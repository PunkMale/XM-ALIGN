import argparse, glob, os, torch, time
from tools import *
from trainer import *
from dataLoader import *

def main(configs):
    parser = argparse.ArgumentParser(description = "AV speaker recognition on VoxCeleb")

    ### Training setting
    parser.add_argument('--configs',                        default=configs)
    # parser.add_argument('--max_epoch',       type=int,      default=configs['max_epoch'])
    parser.add_argument('--max_epoch',       type=int,      default=10)
    parser.add_argument('--batch_size',      type=int,      default=128,      help='Batch size')
    parser.add_argument('--frame_len',       type=int,      default=200,      help='Input length of utterance')
    parser.add_argument('--n_cpu',           type=int,      default=8,       help='Number of loader threads')
    parser.add_argument('--save_step',       type=int,      default=configs.get('save_step', 10),  help='Test and save every [test_step] epochs')
    parser.add_argument('--lr',              type=float,    default=configs['lr'],                 help='Learning rate')
    parser.add_argument("--lr_decay",        type=float,    default=configs['lr_decay'],           help='Learning rate decay every [test_step] epochs')
    parser.add_argument('--augment',         type=bool,     default=configs.get('augment', False))
    parser.add_argument('--loss_type',       type=str,      default=configs.get('loss_type', 'ce'))

    ### Data path
    parser.add_argument('--data_type',    type=str,      default=configs.get('data_type', 'v1'))
    parser.add_argument('--train_list',      type=str,      default=configs['train_list'])
    parser.add_argument('--train_path',      type=str,      default=configs['train_path'])
    parser.add_argument('--random_select',   type=bool,     default=configs.get('random_select', False))
    parser.add_argument('--mav_root',        type=str,      default=configs['mav_root'])
    parser.add_argument('--mav_heard_list',  type=str,      default=configs['mav_heard_list'])
    parser.add_argument('--mav_unheard_list',type=str,      default=configs['mav_unheard_list'])
    parser.add_argument('--musan_path',      type=str,      default="data/musan")
    parser.add_argument('--rir_path',        type=str,      default="data/RIRS_NOISES/simulated_rirs")
    parser.add_argument('--save_path',       type=str,      default=configs['save_path'])
    parser.add_argument('--log_path',        type=str,      default=os.path.join(configs['save_path'], 'train.log'))

    ### Initial modal path
    parser.add_argument('--initial_model_a', type=str,      default=configs.get('initial_model_a', ''))
    parser.add_argument('--initial_model_v', type=str,      default=configs.get('initial_model_v', ''))

    ### Model & loss setting
    parser.add_argument('--embedding_dim_a', type=int,      default=configs.get('embedding_dim_a', 192),   help='Embedding dimmension for audio training')
    parser.add_argument('--margin_a',        type=float,    default=configs.get('margin_a', 0.2),          help='AAM Loss margin for audio training')
    parser.add_argument('--scale_a',         type=float,    default=configs.get('scale_a', 30),            help='AAM Loss scale for audio training')

    parser.add_argument('--embedding_dim_v', type=int,      default=configs.get('embedding_dim_v', 512),   help='Embedding dimmension for visual training')
    parser.add_argument('--margin_v',        type=float,    default=configs.get('margin_v', 0.2),          help='AAM Loss margin for visual training')
    parser.add_argument('--scale_v',         type=float,    default=configs.get('scale_v', 64),            help='AAM Loss scale for visual training')
    
    # Embedding Alignment
    parser.add_argument('--embedding_alignment',  type=bool,  default=configs.get('embedding_alignment', False))
    parser.add_argument('--alignment_loss',       type=str,   default=configs.get('alignment_loss', 'mse'))
    parser.add_argument('--alignment_weight',     type=float, default=configs.get('alignment_weight', 0))

    # Share Clissifier
    parser.add_argument('--share_head',           type=bool,  default=configs.get('share_head', False))
    
    ## Init folders
    args = init_system(parser.parse_args())
    args.logger.print(f"{args}")
    ## Init loader
    args = init_train_loader(args)
    ## Init trainer
    s = init_trainer(args)

    while args.epoch <= args.max_epoch:
        # training
        s.train_network(args)
        # save model
        if args.epoch % args.save_step == 0 or args.max_epoch - args.epoch <= 5:
            s.save_parameters(args.model_save_path_a + "/model_%04d.model"%args.epoch, 'A')
            s.save_parameters(args.model_save_path_v + "/model_%04d.model"%args.epoch, 'V')
        args.epoch += 1
        
    # model averaging
    if not os.path.exists(os.path.join(args.model_save_path_a, "model_avg.model")) or not os.path.exists(os.path.join(args.model_save_path_v, "model_avg.model")):
        model_paths_a = []
        model_paths_v = []
        for i in range(5):
            path_a = args.model_save_path_a + "/model_%04d.model"%(args.max_epoch-i)
            path_v = args.model_save_path_v + "/model_%04d.model"%(args.max_epoch-i)
            model_paths_a.append(path_a)
            model_paths_v.append(path_v)
        s.load_averaged_parameters(model_paths_a, 'A')
        s.load_averaged_parameters(model_paths_v, 'V')
        s.save_parameters(args.model_save_path_a + "/model_avg.model", 'A')
        s.save_parameters(args.model_save_path_v + "/model_avg.model", 'V')
    
    # testing
    s.eval_mav_network(args, heard=True)
    s.eval_mav_network(args, heard=False)

    args.logger.print('Training is finished!')
    return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    config_files = [
        # 'conf/#2_v1_aam_alignment_0.1_english.yaml',
        # 'conf/#2_v1_aam_alignment_0.1_urdu.yaml',
        # 'conf/#3_v1_ce_alignment_cosine_1.0_english.yaml',
        # 'conf/#6_v1_share_ce_english.yaml',
        # 'conf/#7_v1_share_ce_alignment_0.1_english.yaml',
        'conf/#10_v3_ce_alignment_0.1_english.yaml',
        'conf/#10_v3_ce_alignment_0.1_german.yaml',
        'conf/#13_v3_share_ce_alignment_0.1_english.yaml'
    ]
    for config_file in config_files:
        configs = parse_config_or_kwargs(config_file)
        main(configs)
