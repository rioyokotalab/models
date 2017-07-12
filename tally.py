# -*- coding: utf-8 -*-

"""
要件定義
ファイルのディレクトリをpathで渡す
そのディレクトリ内の 000.log~099.logまでを読み込み

inference time: 2.45716309547(sec)
processing time: 4.70762586594(sec)

みたいなとこだけ抜き出す

すべての値を足し合わせて平均をとって出力
"""

def output_average_time(path):
    sum_inference_time = 0
    sum_processing_time = 0
    for n in range(0, 100):
        file_name = path+'/logs/%03d.log' % (n)
        with open(file_name,'r') as f:
            for line in f:
                inference_index=line.find("inference time:")
                if inference_index != -1:
                    split_list = line.split()
                    tmpstring = split_list[2].strip('(sec)')
                    sum_inference_time += float(tmpstring)
                processing_index=line.find("processing time:")
                if processing_index != -1:
                    split_list = line.split()
                    tmpstring = split_list[2].strip('(sec)')
                    sum_processing_time += float(tmpstring)

    average_inference_time = sum_inference_time/100
    average_processing_time = sum_processing_time/100
    print("average_inference_time :",average_inference_time)
    print("average_processing_time :",average_processing_time)

print("[squeezenet]")
output_average_time("squeezenet")

print("[bvlc_googlenet]")
output_average_time("bvlc_googlenet")
