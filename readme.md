DELSYS肌电数据计算工具
==



目录
--

1. [DELSYS肌电数据介绍](#DELSYS肌电数据介绍)
2. [EMG基本数据计算介绍](#EMG基本数据计算介绍)
3. [EMG数据处理步骤](#EMG数据处理步骤)



## DELSYS肌电数据介绍

* delsys肌电采集的原始数据是```.hpf```类型的数据，在官方的肌电分析软件中可以进行很多数据的分析，该种方法在小样本的数据处理中效果不错，但是，当数据量很大时需要导出原始数据进行分析。

* delsys官方提供了批量导出数据工具```delsys File Utility```

  ![Delsys-file-conversion](./Delsys-file-conversion.png)

* 在使用官方工具导出数据时，一般批量选择一个文件夹，不需要勾选```Output headers```，一般导出csv文件
* 导出的```.csv```类型的数据文件格式一般就是一个表头和批量的数据，表头是时间序列加上每块肌肉的EMG值和ACC值。

## EMG基本数据计算介绍

1. 拿到归一化的数据，首先进行去漂步骤，remove mean，所有数据减去平均值（当有ACC数据的时候，需要把ACC数据排除在外）

2. 寻找相关的周期，进行阶段的划分，根据运动学或者动力学等数据进行划分（或者可以直接根据肌电的数据进行阶段的划分）

3. 进行某些阶段数据的计算，包括常规的RMS、iEMG等；进行傅里叶变化之后的MPF，MF等数据

   * 补充知识：

   * > 频谱通常既含有幅度也含有相位信息；**幅度谱的平方（二次量纲）又叫能量谱（密度）**，它描述了信号能量的频域分布；功率信号的功率谱（密度）描述了信号功率随频率的分布特点（密度：单位频率上的功率），业已证明，平稳信号功率谱密度恰好是其自相关函数的傅氏变换。对于非平稳信号，其自相关函数的时间平均（对时间积分，随时变性消失而再次退变成一维函数）与功率谱密度仍是傅氏变换对；
     >
     > 在工程实际中，即便是功率信号，由于持续的时间有限，可以**直接对信号进行傅里叶变换，然后对得到的幅度谱的模求平方，再除以持续时间来估计信号的功率谱**。
     >
     > 通常来说，EMG信号属于功率信号，是周期的信号（一般来说，周期信号和随机信号是功率信号，而非周期的确定信号是能量信号。）
     >
     > * 进行傅里叶变换的时候，选取的点的个数最好保证取到大部分的原来的点，直接//2就可以，不用考虑2的N次幂（可能会丢失很多的点）
     > * 功率谱曲线所覆盖面积在数值上等于信号的总功率（能量能量谱求和）对功率谱在频域上积分就可以得到信号的功率。
     > * 信号幅度平方的积分，如果是数字信号，能量就是各点信号幅度值平方后的求和。
     > * MPF平均功率频率是总功率除以总时间（**可以在功率谱曲线中计算每个频率下功率与频率的乘积，最后取平均值，找到均值对应的那个频率就是所需要求的频率**）
     > * MF中值频率将能量谱的能量一分为二的频率（**可以对能量谱进行积分然后找一半**）
     >
     > 能量谱又称能量谱密度，针对能量信号，是指用密度的概念表示信号能量在各频率点的分布情况。也就是说，在**能量谱中对频率进行积分，就可以得到信号的能量**。能量谱反映了能量信号的能量随频率的变化情况，是**原信号的傅里叶变换绝对值的平方**，单位为焦耳/HZ。
     >
     > ![MF-MPF](./MF-MPF.png)

4. 进行后期数据的更进一步的数据处理，包括小波变换、线性包络线的计算等

   * 补充知识（从傅里叶变换 到小波变换）

     > 对比：傅里叶变换处理非平稳信号有天生缺陷。它只能获取**一段信号总体上包含哪些频率的成分**，但是**对各成分出现的时刻并无所知**。因此时域相差很大的两个信号，可能频谱图一样。
     >
     > 提出STFT短时傅里叶变换（加窗口进行傅里叶变换）：小波变换的出发点和STFT还是不同的。**STFT是给信号加窗，分段做FFT**；而小波直接把傅里叶变换的基给换了——将**无限长的三角函数基**换成了**有限长的会衰减的小波基**。这样**不仅能够获取频率**，还可以**定位到时间**
     >
     > ![小波变换](./小波变换.png)

## EMG数据处理步骤

1. 将原始数据导入，进行归一化处理
2. 将数据的表头进行整理，表头至少需要包含肌肉的名称，（原始的数据header包含了肌肉名称，指标的种类，采样频率，总数据个数，单位）
3. 提出一列时间轴作为第一列，其它的时间轴删除
4. 需要的数据中，提取出EMG的数据，（可以适当加上某些肌肉的ACC数据进行动作阶段的划分）
5. 归一化的数据输入的df中包括第一列的t时间轴，和后面每块肌肉的数据
6. 进行数据可视化
7. 数据储存
