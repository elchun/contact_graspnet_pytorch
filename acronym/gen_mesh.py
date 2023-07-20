import os

def convert(model_dir):
    print(f'Converting {model_dir}')
    i = 0
    for fname in os.listdir(model_dir):
        if '.obj' not in fname:
            continue
        print(fname)
        path = os.path.join(model_dir, fname)
        os.system(f'./Manifold/build/manifold {path} temp.watertight.obj -s')
        os.system(f'./Manifold/build/simplify -i temp.watertight.obj -o {os.path.join(model_dir, "w_" + fname)} -m -r 0.02')
        i += 1
        if i == 10:
            break
    print('Num models: ', i)

if __name__ == '__main__':
    model_dir = './models'
    convert(model_dir)
