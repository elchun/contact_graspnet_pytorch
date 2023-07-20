import trimesh
import os



def repair_mesh(filepath):
    mesh = trimesh.load(filepath, process=False, force='meshfix')
    idxs = trimesh.repair.broken_faces(mesh)
    print(idxs)

if __name__ == '__main__':
    root_dir = './models'
    root_dir = './'
    ids = [
        '49a0a84ee5a91344b11c5cce13f76151',
        'feb146982d0c64dfcbf4f3f04bbad8',
        '202fd2497d2e85f0dd6c14adedcbd4c3',
        '47fc4a2417ca259f894dbff6c6a6be89',
        '2a7d62b731a04f5fa54b9afa882a89ed',
        '8123f469a08a88e7761dc3477e65a72',

        '202fd2497d2e85f0dd6c14adedcbd4c3',
        'feb146982d0c64dfcbf4f3f04bbad8',
        '47fc4a2417ca259f894dbff6c6a6be89',
        'feb146982d0c64dfcbf4f3f04bbad8',
        '2a7d62b731a04f5fa54b9afa882a89ed',
        'b54e412afd42fe6da7c64d6a7060b75b',
    ]

    for id in ids:
        fname = f'{id}.obj'
        fname = f'temp.{id}.watertight.obj'
        file_name = os.path.join(root_dir, fname)
        print(file_name)
        repair_mesh(file_name)