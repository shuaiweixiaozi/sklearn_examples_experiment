import time

stock_dict = {}
with open("D:/stock_jiaoyan/amount/output/AmountMonitoringJob_04-20_04-27.csv", 'r') as f:
    # dick结构：{key:code_date,value:[timestamp,amount_ratio]}
    for line in f.readlines():
        splits = line.split(";")
        key = splits[0] + "_" + splits[1]
        if key not in stock_dict.keys():
            stock_dict[key] = [splits[2], splits[3]]
        else:
            if float(splits[3]) > float(stock_dict[key][1]):
                stock_dict[key] = [splits[2], splits[3]]

with open("D:/stock_jiaoyan/amount/output/AmountFinal_04-20_04-27.csv", 'w') as f:
    for (k, v) in stock_dict.items():
        print(k + ";" + v[0] + ";" + v[1])
        time_stamp = int(v[0]) / 1000
        time_array = time.localtime(time_stamp)
        time_str = time.strftime('%H:%M:%S', time_array)
        f.writelines(k + "," + v[0] + "," + v[1][0:-1] + "," + time_str + "\n")
