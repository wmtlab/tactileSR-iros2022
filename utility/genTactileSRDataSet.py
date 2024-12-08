import os, sys
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    from .RegTactileData import RegTactileData
except:
    from RegTactileData import RegTactileData

"""
convert raw_data to TSR_data
----------------------> [0/x]
\  0 1 2 3 4 5 6 7 8 9
0  - - - - - - - - - - 
1  - - - - - - - - - - 
2  - - * * * * * * - - 
3  - - * * * * * * - - 
4  - - * * * * * * - - 
5  - - * * * * * * - - 
6  - - * * * * * * - - 
7  - - * * * * * * - - 
8  - - - - - - - - - - 
9  - - - - - - - - - - 
[2,8) [2,8) 36
[3,7)       16
[4,6)       4
"""

def delFile(path):
    for root, ds, fs, in os.walk(path):
        for f in fs:
            os.remove(path+'/'+f)
            print("remove file:%s ..." % f)

    
def genDataFunc(raw_data_file, tsr_data_path, ProcessData, file_num, res=10, train_ratio=80, travl_num = (4,6)):
    if not os.path.exists(tsr_data_path+'train/'):
        os.makedirs(tsr_data_path+'train/')
    if not os.path.exists(tsr_data_path+'test/'):
        os.makedirs(tsr_data_path+'test/')

    train_num, test_num = 0, 0
    obj_name, suffix_dot = os.path.splitext(raw_data_file)
    data_type = obj_name.split('/')[-1].split('_')[0]

    data_x, data_y, data_z = ProcessData.readData(raw_data_file)
    data_x, data_y, data_z = ProcessData.thresholdFilter(data_x, data_y, data_z,thresholdScale=0.95)
    data_x, data_y, data_z = ProcessData.TactileSeq2Single(data_x), ProcessData.TactileSeq2Single(data_y), ProcessData.TactileSeq2Single(data_z)
    data_x, data_y, data_z = ProcessData.scalePattern(data_x, data_y, data_z)
    if res == 10:
        if data_type == '2':
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_new(data_x), ProcessData.regData_new(data_y), ProcessData.regData_new(data_z)
        else:
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData(data_x), ProcessData.regData(data_y), ProcessData.regData(data_z)
    elif res == 5:
        if data_type == '2':
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_new_5(data_x), ProcessData.regData_new_5(data_y), ProcessData.regData_new_5(data_z)
        else:
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_5(data_x), ProcessData.regData_5(data_y), ProcessData.regData_5(data_z)
    elif res == 2:
        if data_type == '2':
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_new_2(data_x), ProcessData.regData_new_2(data_y), ProcessData.regData_new_2(data_z)
        else:
            sr_data_x, sr_data_y, sr_data_z = ProcessData.regData_2(data_x), ProcessData.regData_2(data_y), ProcessData.regData_2(data_z)
    else:
        sys.exit()

    sr_data_x, sr_data_y, sr_data_z = ProcessData.smoothPattern(sr_data_x, sr_data_y, sr_data_z, smoth_method=1)
    
    for i in range(travl_num[0], travl_num[1]):
        for j in range(travl_num[0], travl_num[1]):
            LR_data_x = data_x[i,j].reshape(4,4)
            LR_data_y = data_y[i,j].reshape(4,4)
            LR_data_z = data_z[i,j].reshape(4,4)
            LR_data = np.array([LR_data_x, LR_data_y, LR_data_z])

            HR_data_x = sr_data_x
            HR_data_y = sr_data_y
            HR_data_z = sr_data_z
            HR_data = np.array([HR_data_x, HR_data_y, HR_data_z])

            dataset = {'LR':LR_data, 'HR':HR_data}
            save_name = str(file_num) + '_' + str(i)+str(j) + '_'+'00'+'.npy'
            if random.randint(0,100) < train_ratio:
                np.save(file=tsr_data_path +'train/'+ save_name, arr=dataset, allow_pickle=True)
                train_num += 1
            else:
                np.save(file=tsr_data_path +'test/'+ save_name, arr=dataset, allow_pickle=True)
                test_num += 1

            # mirror
            LR_data_x_mir = np.fliplr(LR_data_x)
            LR_data_y_mir = np.fliplr(LR_data_y)
            LR_data_z_mir = np.fliplr(LR_data_z)
            LR_data = np.array([LR_data_x_mir, LR_data_y_mir, LR_data_z_mir])

            HR_data_x_mir = np.fliplr(HR_data_x)
            HR_data_y_mir = np.fliplr(HR_data_y)
            HR_data_z_mir = np.fliplr(HR_data_z)
            HR_data = np.array([HR_data_x_mir, HR_data_y_mir, HR_data_z_mir])
            save_name = str(file_num) + '_' + str(i) + str(j) + '_' + '10'+'.npy'
            dataset = {'LR':LR_data, 'HR':HR_data}
            if random.randint(0,100) < train_ratio:
                np.save(file=tsr_data_path +'train/'+ save_name, arr=dataset, allow_pickle=True)
                train_num += 1
            else:
                np.save(file=tsr_data_path +'test/'+ save_name, arr=dataset, allow_pickle=True)
                test_num += 1

            # rotate
            for rot_index in range(1,4):
                # original image rotate
                LR_data_rot = np.array([np.rot90(LR_data_x, rot_index), 
                                        np.rot90(LR_data_y, rot_index), 
                                        np.rot90(LR_data_z, rot_index)])
                
                HR_data_rot = np.array([np.rot90(HR_data_x, rot_index), 
                                        np.rot90(HR_data_y, rot_index), 
                                        np.rot90(HR_data_z, rot_index)])
                
                save_name = str(file_num) + '_' + str(i) + str(j) + '_' + '0'+str(rot_index)+'.npy'
                dataset = {'LR':LR_data_rot, 'HR':HR_data_rot}  
                if random.randint(0,100) < train_ratio:
                    np.save(file=tsr_data_path +'train/'+ save_name, arr=dataset, allow_pickle=True)
                    train_num += 1
                else:
                    np.save(file=tsr_data_path +'test/'+ save_name, arr=dataset, allow_pickle=True)
                    test_num += 1


                # mirror image rotate
                LR_data_mir = np.array([np.rot90(LR_data_x_mir, rot_index), 
                                        np.rot90(LR_data_y_mir, rot_index), 
                                        np.rot90(LR_data_z_mir, rot_index)])
                HR_data_mir = np.array([np.rot90(HR_data_x_mir, rot_index), 
                                        np.rot90(HR_data_y_mir, rot_index), 
                                        np.rot90(HR_data_z_mir, rot_index)])
                
                save_name = str(file_num) + '_' + str(i) + str(j) + '_' + '1'+str(rot_index)+'.npy'
                dataset = {'LR':LR_data_mir, 'HR':HR_data_mir}  
                if random.randint(0,100) < train_ratio:
                    np.save(file=tsr_data_path +'train/'+ save_name, arr=dataset, allow_pickle=True)
                    train_num += 1
                else:
                    np.save(file=tsr_data_path +'test/'+ save_name, arr=dataset, allow_pickle=True)
                    test_num += 1

    print(raw_data_file, ", save_success.. train_num = {}, test_num = {} ". format(train_num, test_num))


def genDataset(raw_data_path, tsr_data_path, ProcessData, file_num, res=10, train_ratio=80, travl_num = (4,6)):
    ## ---- clear folder ---- ## 
    delFile(tsr_data_path+'train/')
    delFile(tsr_data_path+'test/')

    # ---- generate data ---- ## 
    file_num = 0
    for root, ds, fs, in os.walk(raw_data_path):
        for f in fs:
            obj_name, suffix_dot = os.path.splitext(f)
            if suffix_dot == '.npy':
                data_type = obj_name.split('/')[-1].split('_')[0]
                ## --- all data --- ##
                # genData(raw_data_path+f, tsr_data_path, ProcessData, file_num=file_num)

                ## --- old data --- ##
                # if not data_type == '2':
                    # genData(raw_data_path+f, tsr_data_path, ProcessData, file_num=file_num, res=res)
                    # file_num += 1
                
                ## --- old data + part new data --- ##
                if data_type == '2':
                    ## --- only letter --- ##
                    data_name = obj_name.split('/')[-1].split('_')[1]
                    if len(data_name) == 1:
                        genDataFunc(raw_data_path+f, tsr_data_path, ProcessData, file_num=file_num, res=res, train_ratio=train_ratio, travl_num=travl_num)
                        file_num += 1
                    else:
                        pass
                else:
                    genDataFunc(raw_data_path+f, tsr_data_path, ProcessData, file_num=file_num, res=res, train_ratio=train_ratio, travl_num=travl_num)
                    file_num += 1 


def visData(data_x, data_y, data_z, 
            data_x_hr, data_y_hr, data_z_hr):
    
    xy_vmin, xy_vmax = -0.5, 0.5
    z_vmin, z_vmax = 0, 2

    cmap = 'winter'

    fig = plt.figure()
    ax_1 = fig.add_subplot(231)
    ax_2 = fig.add_subplot(232)
    ax_3 = fig.add_subplot(233)

    ax_4 = fig.add_subplot(234)
    ax_5 = fig.add_subplot(235)
    ax_6 = fig.add_subplot(236)

    ax_1.imshow(data_x,vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
    ax_2.imshow(data_y,vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
    ax_3.imshow(data_z,vmin=z_vmin, vmax=z_vmax, cmap=cmap)

    ax_4.imshow(data_x_hr,vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
    ax_5.imshow(data_y_hr,vmin=xy_vmin, vmax=xy_vmax, cmap=cmap)
    ax_6.imshow(data_z_hr,vmin=z_vmin, vmax=z_vmax, cmap=cmap)

    ax_1.axis('off')
    ax_2.axis('off')
    ax_3.axis('off')
    ax_4.axis('off')
    ax_5.axis('off')
    ax_6.axis('off')

    ax_1.set_title('LR_X axis')
    ax_2.set_title('LR_Y axis')
    ax_3.set_title('LR_Z axis')
    ax_4.set_title('HR_X axis')
    ax_5.set_title('HR_Y axis')
    ax_6.set_title('HR_Z axis')

    plt.savefig('./out.png')
    plt.close()


if __name__ == "__main__":

    res = 10
    train_ratio = 80
    travl_num = (4,6)

    dirname, _ = os.path.split(os.path.abspath(sys.argv[0]))
    root_path = os.path.dirname(dirname) + '/'
    raw_data_path = root_path + 'dataset/raw_data/'
    tsr_data_path = root_path + f'dataset/TSR_data_x{res}/'
    ProcessData = RegTactileData()
    
    genDataset(raw_data_path, tsr_data_path, ProcessData, file_num=0, res=res, train_ratio=train_ratio, travl_num=travl_num)

    ## ---- test case ---- ##
    test_file_name = 'train/3_45_12.npy'
    data = np.load(tsr_data_path+test_file_name, allow_pickle=True).item()
    lr_data, hr_data = data['LR'], data['HR']
    print(hr_data.shape)
    print(lr_data.shape)
    visData(lr_data[0], lr_data[1], lr_data[2], 
            hr_data[0], hr_data[1], hr_data[2])



