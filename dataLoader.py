import glob, numpy, os, random, soundfile, torch, cv2, wave
from tqdm import tqdm
from scipy import signal
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as T

def init_train_loader(args):
    train_dataset = MavCeleb(**vars(args))
    args.n_class = train_dataset.get_speaker_number()
    args.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True, pin_memory = True)
    return args

def get_mavceleb_list(train_list, train_path):
    wav_data_list = []
    img_data_list = []
    label_list = []
    audio_visual_dict = {}
    lines = open(train_list).read().splitlines()
    
    for idx, line in tqdm(enumerate(lines), desc='Read MAV-Celeb list', ncols=120, total=len(lines)):
        wav_label = line.split()[0].split('/')[0]
        img_label = line.split()[1].split('/')[0]
        if wav_label != img_label:
            print(f"line {idx} label not match: {wav_label} {img_label}")
        wav_file_path = os.path.join(train_path, 'voices', line.split()[0])
        img_file_path = os.path.join(train_path, 'faces', line.split()[1])
        wav_data_list.append(wav_file_path)
        img_data_list.append(img_file_path)
        label_list.append(wav_label)

    return wav_data_list, img_data_list, label_list


class MavCeleb(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, frame_len, augment=False, **kwargs):
        self.train_path = train_path
        self.num_frames = frame_len
        ##################### noise list #####################
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
        self.noiselist = {}
        self.augment = augment
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        for file in augment_files:
            if file.split('/')[-3] not in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
        print(f"musan_noise_files:{len(self.noiselist['noise'])}")
        print(f"musan_speech_files:{len(self.noiselist['speech'])}")
        print(f"musan_music_files:{len(self.noiselist['music'])}")
        self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
        print(f"rir_files:{len(self.rir_files)}")

        # read MAV-Celeb list
        mav_wav_data_list, mav_img_data_list, mav_label_list = get_mavceleb_list(train_list, train_path)

        self.wav_data_list = mav_wav_data_list
        self.img_data_list = mav_img_data_list
        self.label_list = mav_label_list

        # label: id -> int
        unique_ids = sorted(set(self.label_list))  # 自动去重并排序（等价于 list(set(...)) + sort()
        str_to_int_label = {key: idx for idx, key in enumerate(unique_ids)}
        self.int_label_list = [str_to_int_label[label] for label in self.label_list]

        print('speaker number:{}'.format(self.get_speaker_number()))
        print('training data number:{}'.format(len(self.int_label_list)))

    def __getitem__(self, index):
        label = self.int_label_list[index]

        wav_file = self.wav_data_list[index]
        audio = self.load_wav(wav_file)

        img_file = self.img_data_list[index]
        image = self.load_face(img_file)

        return audio, image, label

    def load_wav(self, file):
        try:
            audio, sr = soundfile.read(file)
        except:
            print(f'read error: {file}')
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)

        # add noise
        if self.augment:
            augtype = random.randint(1,4)
            if augtype == 1:
                aug_audio = self.add_rev(audio, length=length)
            elif augtype == 2:
                aug_audio = self.add_noise(audio, 'music', length=length)
            elif augtype == 3:
                aug_audio = self.add_noise(audio, 'speech', length=length)
            elif augtype == 4:
                aug_audio = self.add_noise(audio, 'noise', length=length)
            return torch.FloatTensor(aug_audio[0])
        else:
            return torch.FloatTensor(audio[0])

    def load_face(self, file):
        try:
            frame = cv2.imread(file)
        except:
            print(f'read error: {file}')
        face = cv2.resize(frame, (112, 112))
        face = self.face_aug(face)
        return face

    def __len__(self):
        return len(self.wav_data_list)
    
    def get_speaker_number(self):
        return max(self.int_label_list) + 1

    def face_aug(self, face):		
        global_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9),sigma=(0.1, 5)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
        ])
        return global_transform(face)

    def add_rev(self, audio, length):
        rir_file    = random.choice(self.rir_files)
        rir, sr     = soundfile.read(rir_file)
        rir         = numpy.expand_dims(rir.astype(numpy.float32),0)
        rir         = rir / numpy.sqrt(numpy.sum(rir**2))
        return signal.convolve(audio, rir, mode='full')[:,:length]

    def add_noise(self, audio, noisecat, length):
        clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
        numnoise    = self.numnoise[noisecat]
        noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiselength = wave.open(noise, 'rb').getnframes()
            if noiselength <= length:
                noiseaudio, _ = soundfile.read(noise)
                noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
            else:
                start_frame = numpy.int64(random.random()*(noiselength-length))
                noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
            noiseaudio = numpy.stack([noiseaudio],axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
            noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
        return noise + audio
