# delsys肌电处理程序

# 导入相关的库
import os
import re
import pandas as pd
from scipy import integrate
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor # 用于多线程加速计算


# 获取文件夹中所有类型文件，存入字典中，可以按需要寻找相应类型的文件
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

# 对单个原始csv文件进行处理，这里的原始csv文件直接从delsys工具中导出，包含了表头
def get_one_data(filepath):
    df_head = pd.read_csv(filepath ,encoding='utf-8' ,error_bad_lines=False ,warn_bad_lines=False
                          ,header=None) # read the head
    head_len = len(df_head)
    df_data = pd.read_csv(filepath ,encoding='utf-8' ,skiprows=head_len) # read the data ，这里将表表头去掉了
    t = df_data.iloc[: ,0]
    df_data.drop(labels=0 ,axis=0 ,inplace=True  )# 删除第一行的0值
    df_data.reset_index(drop=True ,inplace=True) # 重新整理索引

    # get the new head
    real_head = df_data.columns # the head we need, to do maybe cut the head str which wo can read
    # 对表头进行清洗，归一化展示
    # 'Label: 2左股直 10: EMG 10 (IM) Sampling frequency: 2.000000e+003 Number of points: 58000 start: 0.000000e+000 Unit: Volts Domain Unit: s'
    ex =  '.* EMG'
    new_head = []
    for i in real_head:
        if 'EMG' in i:
            s = re.match(ex ,i).group()
            new_head.append(i)
    for i in df_data:
        if 'EMG' not in i:
            df_data.drop(labels=i ,axis=1 ,inplace=True)

    df_data.columns = new_head
    df_data.insert(0 ,'t' ,t)

    return df_data ,t

# 该函数和上面的函数一样，这里也是直接对delsys导出的csv数据进行处理，这里导出的肌电数据没有包含表头，这个是较新的好的方法
def get_one_data2(filepath):
    df_data = pd.read_csv(filepath ,encoding='utf-8') # read the data
    t = df_data.iloc[:,0]
    # get the new head
    real_head = df_data.columns # the head we need, to do maybe cut the head str which wo can read
    # 对表头进行清洗，归一化展示
    # 'Label: 2左股直 10: EMG 10 (IM) Sampling frequency: 2.000000e+003 Number of points: 58000 start: 0.000000e+000 Unit: Volts Domain Unit: s'
    ex =  '.* EMG'
    new_head = []
    for i in real_head:
        if 'EMG' in i:
            s = re.match(ex ,i).group()
            new_head.append(i)
    for i in df_data:
        if 'EMG' not in i:
            df_data.drop(labels=i ,axis=1 ,inplace=True)

    df_data.columns = new_head
    df_data.insert(0 ,'t' ,t)
    
    # 在这里进行去平均值，remove mean
    df_data = df_data - df_data.mean()
    
    return df_data ,t


# 肌电常用的带通滤波方法，可以根据具体的研究修改相应的参数
def bandpassfilter(sEMG_df, fre=2000, N=4, low=25, high=400):  # the bandpass filter
    '''bandpass of butterworth filter'''
    # the data is a sequence of one data

    EMGs_columns = sEMG_df.columns
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
        sEMG_df[EMGs_columns[i]] = fildata_ser

    return sEMG_df


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

    for i in range(1, len(sEMG_df.columns)):  # 从1开始为了去掉时间轴
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


def fft_change(data, Fs=2000):
    '''
    对一段信号进行快速傅里叶变换，得到其功率谱，能量谱，振幅谱
    :param data: 一段离散的信号值，array,...
    :param Fs: 信号的采样频率，默认是2000Hz
    :return: 返回这段信号的fx频率横坐标；A1单边振幅谱；P1单边能量谱；psd_values功率谱（直接周期法计算）
    '''
    N = len(data)
    fx = np.linspace(0.0, Fs / 2, N // 2)  # 横坐标（相当于频率）
    #     f = Fs*np.arange(1,N/2)/N
    A2 = np.fft.fft(data)  # 双边傅里叶变换振幅值(使用abs取模长)
    A1 = 2.0 / N * np.abs(A2[0:N // 2])  # 振幅谱,单边

    # power
    P1 = A1 ** 2  # 能量谱power

    # power spectrum 直接周期法计算
    psd_values = P1 / N  # 功率谱（不除以N表示能量谱）

    # power spectrum using correlate（这是别人使用自相关方法计算的，计算EMG直接法就够用了）
    cor_x = np.correlate(data, data, 'same')  # 自相关
    cor_X = np.fft.fft(cor_x, N)
    ps_cor = np.abs(cor_X)
    ps_cor_values = 10 * np.log10(ps_cor[0:N // 2] / np.max(ps_cor))

    return fx, A1, P1, psd_values


def get_fdata(fx, P1, psd_values, Fs=2000):
    '''
    对一段信号进行频域分析，主要获得其中心频率和平均功率频率的指标
    :param P1:能量谱
    :param psd_values:功率谱
    :return 返回MPF和MF
    '''
    # get mean frequency
    mpf_inter = integrate.cumtrapz(fx * psd_values) / integrate.cumtrapz(fx)
    mean_mi = mpf_inter.mean()
    mpf_index = np.where(mpf_inter > mean_mi)[0][0]

    mf_inter = integrate.cumtrapz(P1)  # 对能量谱进行积分，寻找能量谱的二分之一处的频率
    mf_per = mf_inter / mf_inter[-1]
    mf_index = np.where(mf_per > 0.5)[0][0]

    MPF = mpf_index * Fs / len(psd_values)
    MF = mf_index * Fs / len(psd_values)

    return MPF, MF


def find_cycle(one_edata, pfpath=None, slip=125):
    '''寻找某一块肌肉的周期;one_edata是df'''
    # 使用scipy中find_peaks函数找到相关的点
    mean_edata = one_edata.mean()
    peaks, arggs = signal.find_peaks(one_edata, height=mean_edata * 1.75, distance=30)

    # 进行数据的可视化展示
    fig = plt.figure()
    plt.plot(one_edata)
    plt.plot(peaks * slip, arggs['peak_heights'], 'x', color='r')

    # 先找到第一个和最后一个的peak
    first_p = peaks[0]
    last_p = peaks[-1]
    # 起始点和结束点需要再开始和结束点上加上或者减去一个特定的值，这个值一般根据实际情况定（复杂点可以直接用n%激活程度的值寻找）
    d_cho = 30  # rms中的，实际需要乘一个slip滑动计算的距离
    start_time = first_p - d_cho
    end_time = last_p + d_cho

    # 进行全部数据的截取
    cut_emg_df = one_edata.loc[start_time * slip:end_time * slip]
    # 寻找波谷点
    low_ps = []
    for p in range(len(peaks)):
        sp = peaks[p] * slip
        if p == 0:  # 这里会导致第二个波谷被错过添加
            low_p0 = cut_emg_df[cut_emg_df.values == cut_emg_df.loc[:sp].min()].index[0] / slip
            low_ps.append(low_p0)
            plt.axvline(x=low_p0 * slip, color='r', linestyle=':')
            rp = peaks[p + 1] * slip
            low_p1 = cut_emg_df[cut_emg_df.values == cut_emg_df.loc[sp:rp].min()].index[0] / slip
            low_ps.append(low_p1)
            plt.axvline(x=low_p1 * slip, color='r', linestyle=':')
        elif p == len(peaks) - 1:
            low_p = cut_emg_df[cut_emg_df.values == cut_emg_df.loc[sp:].min()].index[0] / slip
            low_ps.append(low_p)
            plt.axvline(x=low_p * slip, color='r', linestyle=':')
        else:
            rp = peaks[p + 1] * slip
            low_p = cut_emg_df[cut_emg_df.values == cut_emg_df.loc[sp:rp].min()].index[0] / slip
            low_ps.append(low_p)
            plt.axvline(x=low_p * slip, color='r', linestyle=':')
    plt.xlabel(str(np.array(low_ps) * slip), fontsize=10)
    if pfpath is not None:  # 进行文件的存储
        fig.savefig(pfpath)

    return low_ps


# 进入一个周期的数据函数,获取最终的结果
def get_oc_emg(raw_df, rms_df, raw_cll):
    '''输入的是一个包含原始数据或者rms数据的字典'''
    iEMG_dic = {}
    RMS_mean_dic = {}
    RMS_cut_dic = {}
    fre_res = {'MPF': {}, 'MF': {}}
    # 循环计算每块肌肉的iEMG
    rnum = 0
    for emg in raw_df:  # 循环每块原始肌电信号数据;属于cpu bound可以使用多进程进行优化
        rnum += 1
        if emg != 't':
            # plt.figure(rnum)
            one_raw_emg = raw_df[emg]
            iEMG_dic[emg] = []
            fre_res['MPF'][emg] = []
            fre_res['MF'][emg] = []
            for i in range(len(raw_cll) - 1):
                si = raw_cll[i]
                ri = raw_cll[i + 1]
                oc_raw_emg = one_raw_emg.loc[si:ri]  # 每一个周期内的原始肌电信号；可以用于后面的时域指标和频域指标的计算
                # plt.plot(oc_raw_emg)
                c_iEMG = integrate.trapz(oc_raw_emg.abs())  # 每个周期内的iEMG,计算之前需要进行翻正
                iEMG_dic[emg].append(c_iEMG)

                # 计算频域指标
                fx, A1, P1, psd_values = fft_change(oc_raw_emg)  # 对每块肌肉的每个周期的肌肉进行傅里叶变换
                mpf, mf = get_fdata(fx, P1, psd_values)  # 计算mpf和mf
                fre_res['MPF'][emg].append(mpf)
                fre_res['MF'][emg].append(mf)

    # 循环计算每块肌肉的RMS
    snum = 0
    for emg in rms_df:  # 循环每块原始肌电信号数据
        if emg != 't':
            # plt.figure(emg)
            one_rms_emg = rms_df[emg]
            RMS_mean_dic[emg] = []
            RMS_cut_dic[emg] = {}
            for i in range(len(raw_cll) - 1):
                RMS_cut_dic[emg][i + 1] = {}
                si = raw_cll[i]
                ri = raw_cll[i + 1]
                oc_rms_emg = one_rms_emg.loc[si:ri]  # 每一个周期内的原始肌电信号
                oc_rms_all = oc_rms_emg.mean()
                # plt.plot(oc_rms_emg)
                RMS_cut_dic[emg][i + 1] = list(oc_rms_emg)
                RMS_mean_dic[emg].append(oc_rms_all)
        snum += 1

    return iEMG_dic, RMS_mean_dic, fre_res, RMS_cut_dic

def save_results(dirpath,selfname,res_df,saindex=False):
    '''
    将结果写入文件中
    :param filepath: 写入的文件路径
    :param res_df: 需要写入的文件，pandas中的DataFrame类型数据格式
    :return: 返回是否写入成功
    '''
    save_status = False
    filepath = os.path.join(dirpath,selfname)
    if len(res_df) != 0:
        res_df.to_csv(filepath,encoding='gbk',index=saindex)
        save_status = True
    else:
        print('文件写入失败-----数据结果空！')

    return save_status


filepath = r'文件路径'
filepath_dic = get_file(filepath)
# for file in filepath_dic['csv']:
# 设置最终的数据存储路径
def jm_path(filepaths):
    for path in filepaths:
        if not os.path.exists(path):
            os.makedirs(path)

all_rms_dir = os.path.join(filepath, 'all_rms')
pfpath_dir = os.path.join(filepath, 'photo')
iEMG_path = os.path.join(filepath, 'iEMG')
rms_mean_path = os.path.join(filepath, 'rms_mean')
fre_path = os.path.join(filepath, 'fre_p')
rms_cut_path = os.path.join(filepath, 'rms_cut')
jm_path([all_rms_dir, pfpath_dir, iEMG_path, rms_mean_path, fre_path, rms_cut_path])

all_cycle_dic = {} # 用于储存所有周期数据的字典，{'name':[cycle]}

def start_cal(file):
    newfilename = os.path.split(file)[1][:-4]  # get the file's name 不包括了文件类型
    dirfilepath = os.path.dirname(file)

    try:

        df_data, t = get_one_data(file) # 获取归一化数据
        filt_df = bandpassfilter(df_data,2000,4,25,400) # 进行滤波
        rms_df = cal_RMS(filt_df) # 计算RMS

        slip = 125 # 设置滑动周期值
        # 先开始寻找周期
        judge_emg = rms_df.iloc[:,2]  # 仅使用一块肌肉寻找到相应的周期即可

        # 寻找相关的周期，并且存储相应的图片
        pfpath = os.path.join(pfpath_dir,newfilename+'_k.png')
        cl_list = find_cycle(judge_emg,pfpath) # 寻找相关的周期， 这里最好将周期进行储存，后面进行导入式的处理（防止由于前面的周期寻找偏差便于后面的处理修改
        raw_cll = np.array(cl_list) * slip  # 按比例缩放到原始数据中的索引位置（以滑动步长进行返回）
        # 将周期数据进行储存，用于后面的修改等操作
        all_cycle_dic[newfilename] = raw_cll

        # 进行周期中的数据计算 /此处的 raw_cll 可以从储存的字典中直接进行索引，也可以通过外部获取到的字典进行索引
        iEMG_dic, RMS_mean_dic, fre_res, RMS_cut_dic= get_oc_emg(filt_df, rms_df, raw_cll)

        iemg_res_df = pd.DataFrame(iEMG_dic)
        rms_mean_df = pd.DataFrame(RMS_mean_dic)

        # 合成最终的fre结果
        df_mpf = pd.DataFrame(fre_res['MPF'])
        df_mf = pd.DataFrame(fre_res['MF'])
        fre_res_fdf = pd.concat({'MPF': df_mpf, 'MF': df_mf})

        # 合成最终的rms阶段数据的结果
        rms_cut_dic = {}
        for mus in RMS_cut_dic:
            dc = RMS_cut_dic[mus]
            df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in dc.items()]))  # 将长度不等的字典中的数据转化成为DataFrame类型数据
            rms_cut_dic[mus] = df
        rms_cut_df = pd.concat(rms_cut_dic)

        # 进行最终的数据存储（IO过程）
        save_results(all_rms_dir, newfilename+'_rms.csv', rms_df)  # 储存rms结果
        save_results(iEMG_path,newfilename+'_iemg.csv',iemg_res_df) # 每块肌肉下每个周期内的积分肌电数据
        save_results(rms_mean_path,newfilename+'_rms_mean.csv',rms_mean_df) # 每块肌肉下每个周期内的RMS平均值数据
        save_results(fre_path,newfilename+'_fre.csv',fre_res_fdf,saindex=True) # 每块肌肉的每个周期的频域指标的数据
        save_results(rms_cut_path,newfilename+'_rms_cut.csv',rms_cut_df,saindex=True) # 每块肌肉下每个周期的RMS总数据

        print('%s----over!' % file)
    except Exception as e:
        print('错误原因',e,file)

with ThreadPoolExecutor() as tpool:
    results = tpool.map(start_cal,filepath_dic['csv'])

# all_cycle_df = pd.DataFrame(all_cycle_dic) # 这里的dic中的周期的长度是不一样的，这里变成DataFrame会报错
# 长度不一样长的字典转化成为DataFrame的方法，如下：
all_cycle_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_cycle_dic.items()]))
cycle_filepath = os.path.join(filepath,'cycle.csv')
all_cycle_df.to_csv(cycle_filepath,encoding='gbk')
# for file in filepath_dic['csv']:
#     start_cal(file)