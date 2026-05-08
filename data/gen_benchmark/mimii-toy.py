import os
import json


class MIMIIToySolver(object):
    CLSNAMES_2D = ['fan', 'pump', 'slider', 'valve']

    def __init__(self, root):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['id_00', 'id_02', 'id_04', 'id_06']
        self.CLSNAMES = self.CLSNAMES_2D

    def run(self):
        info = {phase: {} for phase in self.phases}
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in self.phases:
                cls_info = []
                species = os.listdir(f'{cls_dir}/{phase}')
                for specie in species:
                    is_abnormal = True if specie not in ['normal'] else False
                    img_dir = f'{cls_dir}/{phase}/{specie}/'
                    img_names = os.listdir(img_dir)
                    img_names.sort()
                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{img_dir.replace(cls_dir, cls_name)}{img_name}',
                            mask_path='',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")


# Good! Run me to generate toy meta.json for MIMII dataset. The meta.json will be used by the benchmark.
if __name__ == '__main__':
    runner = MIMIIToySolver(root='data/dcase-2020-spectrogram')
    runner.run()
