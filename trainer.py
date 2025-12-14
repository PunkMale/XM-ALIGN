import torch, sys, os, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tqdm import tqdm
from tools import *
from module.loss import *
from module.audiomodel import *
from module.visualmodel import *
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast, GradScaler


def init_trainer(args):
    s = trainer(args)
    args.epoch = 1
    if args.initial_model_a != '':
        s.load_parameters(args.initial_model_a, 'A')
    elif len(args.modelfiles_a) >= 1:
        args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles_a[-1]))[0][6:]) + 1
        s.load_parameters(args.modelfiles_a[-1], 'A')

    if args.initial_model_v != '':
        s.load_parameters(args.initial_model_v, 'V')
    elif len(args.modelfiles_v) >= 1:
        args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles_v[-1]))[0][6:]) + 1
        s.load_parameters(args.modelfiles_v[-1], 'V')

    return s


class trainer(nn.Module):
    def __init__(self, args):
        super(trainer, self).__init__()
        # 声纹模型和人脸模型
        self.speaker_encoder    = ECAPA_TDNN(embedding_size=args.embedding_dim_a).cuda()
        self.face_encoder       = IResNet(num_features=args.embedding_dim_v).cuda()
        # 分类层+损失函数
        if args.loss_type == 'ce':
            self.speaker_loss = Softmax(n_class = args.n_class, emb_dim = args.embedding_dim_a).cuda()
            self.face_loss    = Softmax(n_class =  args.n_class, emb_dim = args.embedding_dim_v).cuda()
        elif args.loss_type == 'aam':
            self.speaker_loss = AAMsoftmax(n_class = args.n_class, m = args.margin_a, s = args.scale_a, emb_dim = args.embedding_dim_a).cuda()
            self.face_loss    = AAMsoftmax(n_class =  args.n_class, m = args.margin_v, s = args.scale_v, emb_dim = args.embedding_dim_v).cuda()
        else:
            raise ValueError(f"Unknown loss type: {args.loss_type}!")
        
        # Embedding Alignment
        self.use_alignment = args.embedding_alignment
        if self.use_alignment:
            self.alignment_weight = args.alignment_weight
            if args.alignment_loss == 'cosine':
                self.alignment_loss = cosine_similarity_loss
            elif args.alignment_loss == 'mse':
                self.alignment_loss = mse_loss
            else:
                raise ValueError(f"Unknown alignment loss type: {args.alignment_loss}!")
        # 是否共享分类层
        self.share_head   = args.share_head
        # 优化器
        self.optim           = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 5, gamma = args.lr_decay)
        args.logger.print("Speech model para number = %.2fM"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))
        args.logger.print("Face model para number = %.2fM"%(sum(param.numel() for param in self.face_encoder.parameters()) / 1e6))
        

    def train_network(self, args):
        self.scheduler.step(args.epoch - 1)
        self.train()
        scaler = GradScaler()
        index, loss, loss_a, loss_v, loss_align = 0, 0, 0, 0, 0
        total_acc_a, total_acc_v = 0, 0
        lr = self.optim.param_groups[0]['lr']
        time_start = time.time()

        all_alignment_loss = []

        for num, (speech, face, labels) in enumerate(args.train_loader, start = 1):
            self.zero_grad()
            labels = labels.cuda()
            with autocast():
                a_embedding   = self.speaker_encoder.forward(speech.cuda(), aug = True)
                v_embedding   = self.face_encoder.forward(face.cuda())

                # Speaker Loss
                aloss, acc_a = self.speaker_loss.forward(a_embedding, labels)

                # Face Loss
                if self.share_head:
                    vloss, acc_v = self.speaker_loss.forward(v_embedding, labels)
                else:
                    vloss, acc_v = self.face_loss.forward(v_embedding, labels)

                # Embedding Alignment Loss
                if self.use_alignment:
                    align_loss = self.alignment_weight * self.alignment_loss(a_embedding, v_embedding)
                    align_loss = align_loss.mean()
                    total_loss = aloss + vloss + align_loss
                else:
                    total_loss = aloss + vloss

            scaler.scale(total_loss).backward()
            scaler.step(self.optim)
            scaler.update()

            index += 1
            total_acc_a += acc_a
            total_acc_v += acc_v
            loss_a += aloss.detach().cpu().numpy()
            loss_v += vloss.detach().cpu().numpy()
            loss   += total_loss.detach().cpu().numpy()
            time_used = time.time() - time_start

            if self.use_alignment:
                loss_align += align_loss.detach().cpu().numpy()
                message = f"[{args.epoch}] {100 * (num / args.train_loader.__len__()):.2f}% (est {time_used * args.train_loader.__len__() / num / 60:.1f} mins) Lr: {lr:.5f} Loss: {loss/num:.5f} L_A: {loss_a/num:.5f} L_V: {loss_v/num:.5f} L_AV: {loss_align/num:.5f} Acc_A: {acc_a*100:.2f} Acc_V: {acc_v*100:.2f}\r"
            else:
                message = f"[{args.epoch:2d}] {100 * (num / args.train_loader.__len__()):.2f}% (est {time_used * args.train_loader.__len__() / num / 60:.1f} mins) Lr: {lr:.5f}, Loss: {loss/num:.5f} L_A: {loss_a/num:.5f} L_V: {loss_v/num:.5f} Acc_A: {acc_a*100:.2f} Acc_V: {acc_v*100:.2f}\r"
            args.logger.write(message)
            sys.stderr.write(message)
            sys.stderr.flush()
        sys.stdout.write("\n")

        if self.use_alignment:
            message = "Epoch [%d], LR %f, Loss %f, Loss_A %f, Loss_V %f, Loss_Align %f, Acc_A %f, Acc_V %f\n"%(args.epoch, lr, loss/num, loss_a/num, loss_v/num, loss_align/num, total_acc_a*100/index, total_acc_v*100/index)
        else:
            message = "Epoch [%d], LR %f, Loss %f, Loss_A %f, Loss_V %f, Acc_A %f, Acc_V %f\n"%(args.epoch, lr, loss/num, loss_a/num, loss_v/num, total_acc_a*100/index, total_acc_v*100/index)

        args.logger.print(message)
        args.score_file.write(message)
        args.score_file.flush()
        return

    
    def eval_mav_network(self, args, heard=True):
        args.logger.print('Start MAV-Celeb Evaluation...')
        self.eval()

        if heard:
            eval_list = os.path.join('data', f"{args.data_type}_lists", args.mav_heard_list)
        else:
            eval_list = os.path.join('data', f"{args.data_type}_lists", args.mav_unheard_list)
        audio_files = []
        audio_embeddings = {}
        image_files = []
        image_embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            audio_files.append(line.split()[1])
            image_files.append(line.split()[2])
        audio_setfiles = list(set(audio_files))
        image_setfiles = list(set(image_files))
        audio_setfiles.sort()
        image_setfiles.sort()

        # 提取说话人embedding
        for idx, file in tqdm(enumerate(audio_setfiles), desc='extract speaker embedding', total=len(audio_setfiles), ncols=100):
            audio, _ = soundfile.read(os.path.join(args.mav_root, file))
            # Full utterance
            audio = torch.FloatTensor(numpy.stack([audio], axis=0)).cuda()

            # Speaker embedding
            with torch.no_grad():
                audio_embedding = self.speaker_encoder.forward(audio)
                audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
            audio_embeddings[file] = audio_embedding
        
        # 提取人脸embedding
        for idx, file in tqdm(enumerate(image_setfiles), desc='extract face embedding', total=len(audio_setfiles), ncols=100):
            frame = cv2.imread(os.path.join(args.mav_root, file))
            face = cv2.resize(frame, (112, 112))
            face = numpy.array(face)
            face = numpy.transpose(face, (2, 0, 1))
            face = torch.from_numpy(face)
            face = face.float().div_(255)
            face = face.unsqueeze(0)
            with torch.no_grad():
                image_embedding = self.face_encoder.forward(face.cuda())
                image_embedding = F.normalize(image_embedding, p=2, dim=1)
            image_embeddings[file] = image_embedding
        
        # === 开始计算每个样本的 L2 距离并保存结果 ===
        results = []

        args.logger.print('Computing L2 alignment scores...')
        for line in lines:
            id_, audio_path, image_path = line.split()
            a_emb = audio_embeddings[audio_path]    # [1, D]
            i_emb = image_embeddings[image_path]    # [1, D]

            # 计算 L2 距离平方（sum of squared differences）
            l2_distance_sq = F.mse_loss(a_emb, i_emb, reduction='none').sum(dim=1)  # [1]
            # 如果需要开方得到欧氏距离，取消下一行注释
            # l2_distance = torch.sqrt(l2_distance_sq)
            score = l2_distance_sq.item()  # 使用 L2 距离平方作为分数

            results.append(f"{id_} {score:.4f}")

        # === 确定语言和类型以生成文件名 ===
        lang = 'English' if 'English' in eval_list else 'Urdu'
        if 'English' in eval_list:
            lang = 'English'
        elif 'Urdu' in eval_list:
            lang = 'Urdu'
        elif 'German' in eval_list:
            lang = 'German'
        else:
            raise ValueError('Invalid language specified.')

        heard_str = 'heard' if heard else 'unheard'
        filename = f"sub_score_{lang}_{heard_str}.txt"
        save_path = os.path.join(args.submission_save_path, filename)

        # 确保目录存在
        os.makedirs(args.submission_save_path, exist_ok=True)

        # 保存结果
        with open(save_path, 'w') as f:
            f.write('\n'.join(results) + '\n')

        args.logger.print(f"Evaluation scores saved to {save_path}")

    def save_parameters(self, path, modality):
        if modality == 'A':	
            model = OrderedDict(list(self.speaker_encoder.state_dict().items()) + list(self.speaker_loss.state_dict().items()))
        if modality == 'V':
            model = OrderedDict(list(self.face_encoder.state_dict().items()) + list(self.face_loss.state_dict().items()))
        torch.save(model, path)

    def load_parameters(self, path, modality):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        print(f"Load model from: {path}")
        for name, param in loaded_state.items():
            if modality == 'A':
                if ('face_encoder.' not in name) and ('face_loss.' not in name):
                    if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):
                        if name == 'weight':
                            name = 'speaker_loss.' + name
                        else:
                            name = 'speaker_encoder.' + name
                    try:
                        self_state[name].copy_(param)
                    except (KeyError, RuntimeError) as e:
                        if isinstance(e, KeyError):
                            print(f"Skipped: Parameter '{name}' not found in current model.")
                        elif isinstance(e, RuntimeError):
                            expected_shape = self_state[name].shape if name in self_state else "N/A"
                            print(f"Skipped: Shape mismatch for '{name}'. "
                                f"Checkpoint: {param.shape}, Model: {expected_shape}")
            if modality == 'V':
                if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):
                    if ('face_encoder.' not in name) and ('face_loss.' not in name):
                        if name == 'weight':
                            name = 'face_loss.' + name
                        else:
                            name = 'face_encoder.' + name
                    try:
                        self_state[name].copy_(param)
                    except (KeyError, RuntimeError) as e:
                        if isinstance(e, KeyError):
                            print(f"Skipped: Parameter '{name}' not found in current model.")
                        elif isinstance(e, RuntimeError):
                            expected_shape = self_state[name].shape if name in self_state else "N/A"
                            print(f"Skipped: Shape mismatch for '{name}'. "
                                f"Checkpoint: {param.shape}, Model: {expected_shape}")

    def load_averaged_parameters(self, paths, modality):
        assert len(paths) == 5, "The number of model paths must be exactly 5."

        # 初始化一个字典来存储参数的累加值
        accumulated_params = {}
        self_state = self.state_dict()

        # 加载并累加参数
        for path in paths:
            loaded_state = torch.load(path)
            print(f"Load model from: {path}")
            for name, param in loaded_state.items():
                if name not in accumulated_params:
                    accumulated_params[name] = torch.zeros_like(param)
                accumulated_params[name] += param

        # 计算平均参数
        for name, param in accumulated_params.items():
            accumulated_params[name] = param / len(paths)

        # 根据模态选择合适的参数加载方式
        for name, param in accumulated_params.items():
            if modality == 'A':
                if ('face_encoder.' not in name) and ('face_loss.' not in name):
                    if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):
                        if name == 'weight':
                            name = 'speaker_loss.' + name
                        else:
                            name = 'speaker_encoder.' + name
                    try:
                        self_state[name].copy_(param)
                    except (KeyError, RuntimeError) as e:
                        if isinstance(e, KeyError):
                            print(f"Skipped: Parameter '{name}' not found in current model.")
                        elif isinstance(e, RuntimeError):
                            expected_shape = self_state[name].shape if name in self_state else "N/A"
                            print(f"Skipped: Shape mismatch for '{name}'. "
                                f"Checkpoint: {param.shape}, Model: {expected_shape}")
            if modality == 'V':
                if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):
                    if ('face_encoder.' not in name) and ('face_loss.' not in name):
                        if name == 'weight':
                            name = 'face_loss.' + name
                        else:
                            name = 'face_encoder.' + name
                    try:
                        self_state[name].copy_(param)
                    except (KeyError, RuntimeError) as e:
                        if isinstance(e, KeyError):
                            print(f"Skipped: Parameter '{name}' not found in current model.")
                        elif isinstance(e, RuntimeError):
                            expected_shape = self_state[name].shape if name in self_state else "N/A"
                            print(f"Skipped: Shape mismatch for '{name}'. "
                                f"Checkpoint: {param.shape}, Model: {expected_shape}")
