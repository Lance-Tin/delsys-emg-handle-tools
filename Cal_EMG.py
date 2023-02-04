# 主要目的是计算所有的RMS数据，然后存储起来
import os
import re
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from scipy import integrate, signal


def get_file(filepath):
    '''get all the filename from needed dir'''
    filename_dic = {}  # to save all absflie and classify all file , the key is filetype and value is absfilename

    for root, dirs, files in os.walk(filepath):
        for file in files:
            filetype = os.path.splitext(file)[1][1:]
            if filetype not in filename_dic.keys():
                filename_dic[filetype] = []

            filename_dic[filetype].append(os.path.join(root, file))
    print('文件中含有的文件类型有：', filename_dic.keys())
    return filename_dic


def get_one_data2(filepath):
    df_data = pd.read_csv(filepath, encoding='utf-8')  # read the data
    t = df_data.iloc[:, 0]
    # get the new head
    real_head = df_data.columns  # the head we need, to do maybe cut the head str which wo can read
    # 对表头进行清洗，归一化展示
    # 'Label: 2左股直 10: EMG 10 (IM) Sampling frequency: 2.000000e+003 Number of points: 58000 start: 0.000000e+000 Unit: Volts Domain Unit: s'
    ex = '.* EMG'
    new_head = []
    for i in real_head:
        if 'EMG' in i:
            s = re.match(ex, i).group()
            new_head.append(i)
    for i in df_data:
        if 'EMG' not in i:
            df_data.drop(labels=i, axis=1, inplace=True)

    df_data.columns = new_head
    df_data.insert(0, 't', t)

    # 在这里进行去平均值，remove mean
    df_data = df_data - df_data.mean()

    return df_data, t


def bandpassfilter(sEMG_df, fre=2000, N=4, low=25, high=400):  # the bandpass filter
    '''bandpass of butterworth filter'''
    # the data is a sequence of one data

    EMGs_columns = sEMG_df.columns
    new_sEMG_df = pd.DataFrame(columns=EMGs_columns)
    win1 = 2 * low / fre
    win2 = 2 * high / fre

    b, a = signal.butter(N, [win1, win2], 'bandpass')  # 配置滤波器

    for i in range(1, len(sEMG_df.columns)):  # 从1开始为了去掉时间轴
        data = sEMG_df.iloc[::, i].copy()
        data.dropna(axis=0, how='any', inplace=True)  # 删除NaN行
        # 当数据是空的时候需要过滤掉
        # print(len(data))
        if len(data) > 6:
            fildata = signal.filtfilt(b, a, data)  # 进行信号过滤
        else:
            fildata = data
        fildata_ser = pd.Series(fildata)

        #         sEMG_df.iloc[::,i] = fildata # 重新进行赋值 这里会报错，报错的原因就是这里的filedata不是pandas数据类型，那么就要满足隐式索引前后索引匹配
        new_sEMG_df[EMGs_columns[i]] = fildata_ser

    return new_sEMG_df


def cal_RMS(sEMG_df, window=250, slip=125):
    '''
    计算RMS均方根振幅（使用的是滑动窗口的方法进行计算
    :param sEMG_df: 肌电的信号序列
    :param window: 滑动窗口的大小，默认是250帧即0.125
    :param slip: 滑动的步长，默认是125帧，即0.0625
    :return: 返回计算后的emg序列，注意此时的emg相当于有一个缩放，缩放的倍数就是slip的值,后期进行寻找关键帧的时候非常重要
    '''
    # 实际上就是求每个时间下肌电值平方最后求一个平均值
    #     RMS_df = sEMG_analyse.applymap(lambda x: x ** 2).rolling(int(window)).mean().agg(
    #         lambda x: np.sqrt(x)).dropna().iloc[::int(slip), ::] 这里是一个通用的处理方法，但是当数据中有空值的时候，就无法使用了
    # 这里面t时间轴也被求了平方和，可以使用这个进行验证是否正确
    # 采取逐行计算并且重新赋值的方式
    new_EMGdf = sEMG_df[::int(slip)].copy()
    EMGs_columns = new_EMGdf.columns

    for i in range(1, sEMG_df.shape[1]):  # 从1开始为了去掉时间轴
        data = sEMG_df.iloc[::, i].copy()
        data.dropna(axis=0, how='any', inplace=True)  # 删除NaN行
        # 进行过滤，当数据为0的时候无法处理，跳过
        if len(data) != 0:
            RMS_data = data.apply(lambda x: x ** 2).rolling(window).mean().apply(lambda x: np.sqrt(x))[::int(slip)]
            RMS_data.dropna(axis=0, how='any', inplace=True)  # 删除NaN值
            #         RMS_data.reset_index(inplace=True,drop=True)
        else:
            RMS_data = data
        new_EMGdf[EMGs_columns[i]] = RMS_data

    return new_EMGdf


def getres(RMS_ddf, filt_ddf, act_per=0.15):
    # 传入计算之后的RMS数据，然后寻找每块肌肉的最大值，并截取计算平均值和数据输出50ms
    res_all_data_dic = {}
    kf_all_dic = {}
    for d in range(1, len(RMS_ddf.columns)):  # 跳过第一列时间列
        if d == 7:
            continue
        if d == 8:
            d -= 1
        res_allser = pd.Series(dtype='float64')
        kf_allser = pd.Series(dtype='float64')
        mus_ser = RMS_ddf.iloc[:, d].reset_index(drop=True)
        hidden_maxind = np.argmax(mus_ser)
        max_rms = mus_ser.max()
        li = hidden_maxind;
        finl_li = 0  # 左侧索引
        ri = hidden_maxind;
        finl_ri = len(mus_ser) - 1  # 右侧索引
        for mi in range(0, hidden_maxind):
            l_rmsp = RMS_ddf.iloc[li, d] / max_rms
            if l_rmsp < act_per:
                finl_li = li
            else:
                li -= 1
        for mi in range(hidden_maxind, len(mus_ser)):  # 前后循环找10个数，这里设置为将所有的都循环个遍，进行一个寻找
            r_rmsp = RMS_ddf.iloc[ri, d] / max_rms
            if r_rmsp < act_per:
                finl_ri = ri
            else:
                ri += 1
        sta100 = hidden_maxind - 20;
        end100 = hidden_maxind + 20;
        if hidden_maxind < 20:
            sta100 = 0
        if hidden_maxind + 20 > len(mus_ser):
            end100 = len(mus_ser)
        pre_max50_mean = mus_ser[sta100:hidden_maxind].mean()  # 这里修改成100ms（20）
        aft_max50_mean = mus_ser[hidden_maxind + 1:end100].mean()  # 不包括最大的那个值
        act_mean_rms = mus_ser[finl_li:finl_ri].mean()
        pmm_per = 100 * (pre_max50_mean / max_rms)
        amm_per = 100 * (aft_max50_mean / max_rms)
        act_m_per = 100 * (act_mean_rms / max_rms)
        # 计算积分肌电，注意积分肌电的数据长度是需要乘以滑动窗口长度slip的
        pmm50_iEMG = integrate.trapz(filt_ddf.iloc[100 * (sta100):100 * (hidden_maxind), d].abs(), dx=0.0005)
        amm50_iEMG = integrate.trapz(filt_ddf.iloc[100 * (hidden_maxind):100 * (end100), d].abs(), dx=0.0005)
        act_m_iEMG = integrate.trapz(filt_ddf.iloc[100 * (finl_li):100 * (finl_ri), d].abs(), dx=0.0005)

        res_allser['峰值前平均100msRMS'] = pre_max50_mean
        res_allser['峰值后平均100msRMS'] = aft_max50_mean
        res_allser['峰值前后15%RMS'] = act_mean_rms
        res_allser['峰值RMS'] = max_rms
        res_allser['峰值前平均100msRMS百分比'] = pmm_per
        res_allser['峰值后平均100msRMS百分比'] = amm_per
        res_allser['峰值前后15%RMS百分比'] = act_m_per
        res_allser['峰值前平均100msiEMG'] = pmm50_iEMG
        res_allser['峰值后平均100msiEMG'] = amm50_iEMG
        res_allser['峰值前后15%iEMG'] = act_m_iEMG
        res_all_data_dic[muscle_dic[d]] = res_allser

        kf_allser['开始帧'] = finl_li
        kf_allser['结束帧'] = finl_ri
        kf_all_dic[muscle_dic[d]] = kf_allser
    return pd.DataFrame(res_all_data_dic), pd.DataFrame(kf_all_dic)


def plotmaxcur(mus_arr,svfig_fpn,kf_df):
    # 对寻找到的峰值进行一个绘制，图片效果是否可以
    if not os.path.exists(os.path.dirname(svfig_fpn)):
        os.mkdir(os.path.dirname(svfig_fpn))
    count = mus_arr.shape[1]
    for i in range(1,count):
        if i == count-2:
            continue
        if i == count-1:
            i-=1
        sta_kf = kf_df.iloc[0, i - 1]
        end_kf = kf_df.iloc[1, i - 1]
        plt.subplot(7,1,i)
        plt.plot(mus_arr.iloc[:,i].reset_index(drop=True),color=(i**2*0.01, i*0.1, 0.8), label=muscle_dic[i])
        x_max = np.argmax(mus_arr.iloc[:,i])
        plt.axvline(x_max,color='y',linestyle='--',label="峰值位置=100*"+str(x_max))
        plt.axvline(sta_kf,color='r',linestyle='-.',label="左侧15%")
        plt.axvline(end_kf,color='r',linestyle=':',label="右侧15%")
        plt.axvline(x_max-2,color='g',linestyle='--',label="左侧+100帧")
        plt.axvline(x_max+2,color='g',linestyle='--',label="右侧+100帧")
        plt.legend(prop={'family': 'SimHei', 'size': 3},loc=2)
    # plt.show()
    plt.savefig(svfig_fpn,dpi=200)
    plt.close()

def save_results(dirpath, selfname, res_df, saindex=True):
    '''
    将结果写入文件中
    :param filepath: 写入的文件路径
    :param res_df: 需要写入的文件，pandas中的DataFrame类型数据格式
    :return: 返回是否写入成功
    '''
    save_status = False
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    filepath = os.path.join(dirpath, selfname)

    if len(res_df) != 0:
        res_df.to_csv(filepath, encoding='gbk', index=saindex)
        save_status = True
    else:
        logging.info("failed save data! data is empty!")

    return save_status
muscle_dic = {
    1: '左股直肌',
    2: '左股外侧肌',
    3: '左股内侧肌',
    4: '左股二头肌',
    5: '左半腱肌',
    6: '左腓肠肌内侧头',
    7: '左胫骨前肌'
}

def main(count):
    csv_emg_fpn = r''
    save_datares_dirfpn = r''
    save_figres_dirfpn = r''
    need_excel_fpn = r''
    sheet_names = ['Sheet1']
    hd_sheetnms = sheet_names[count] #####################这里是修改处理四个文件的入口，可以多开处理############################
    logging_fpn = r'./jzx_logging_'+hd_sheetnms[:3]+'.log'

    logging.basicConfig(level=logging.DEBUG,  # 设置日志输出格式，仅仅作为记录
                        filename=logging_fpn,  # log日志输出的文件位置和文件名
                        filemode="w",  # 文件的写入格式，w为重新写入文件，默认是追加
                        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                        # 日志输出的格式
                        # -8表示占位符，让输出左对齐，输出长度都为8位
                        datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                        )

    emg_data_dic = get_file(csv_emg_fpn)
    ndf = pd.read_excel(need_excel_fpn,sheet_name=hd_sheetnms)

    save_datares_fpn = os.path.join(save_datares_dirfpn,hd_sheetnms)
    save_figres_fpn = os.path.join(save_figres_dirfpn,hd_sheetnms)

    # for emg_fn in ndf.loc[:,'文件名']:
    for ei in range(0,len(ndf)):
        emg_fn = ndf.loc[ei,'文件名']
        print(emg_fn)
        if ndf.loc[ei,'开始时间'] == np.nan:
            start_kf = int(ndf.loc[ei,'开始时间']*100)
        else:
            start_kf = 0
        if ndf.loc[ei,'结束时间'] == np.nan:
            end_kf = int(ndf.loc[ei,'结束时间']*100)
        else:
            end_kf = 0
        hd_emg_fpn = os.path.join(csv_emg_fpn,hd_sheetnms,emg_fn+'.csv')
        try:
            raw_ddf, t = get_one_data2(hd_emg_fpn)
            if end_kf == 0:
                end_kf = len(raw_ddf)
            ddf = raw_ddf.iloc[start_kf:end_kf,:] # 这里截取一下关键帧
            filt_ddf = bandpassfilter(ddf)
            RMS_ddf = cal_RMS(filt_ddf, window=200, slip=100) # 滑动窗口算法，100帧，相当于是
            plot_svfig_fpn = os.path.join(save_figres_fpn,emg_fn+'.png')
            res_all_ser, afkf_df = getres(RMS_ddf, filt_ddf)
            logging.info("start saving fig ......")
            plotmaxcur(RMS_ddf,plot_svfig_fpn, afkf_df) # 绘制
            save_results(save_datares_fpn,emg_fn+'.csv',res_all_ser)
            print(f"complete======{hd_emg_fpn}")
            logging.info(f"complete======{hd_emg_fpn}")
        except Exception as e:
            logging.exception(e)
            print(f"failed======{hd_emg_fpn}")
            logging.info(f"failed======{hd_emg_fpn}")


if __name__ == "__main__":
    for count in [0]: # ,1,2,3
        main(count)
        print('over!')
