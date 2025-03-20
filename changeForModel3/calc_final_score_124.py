import glob, os
import pandas as pd
import multiprocessing
import warnings

if __name__ == '__main__':
    import opts

    # # /opt/conda/lib/python3.10/multiprocessing/popen_fork.py: 66: RuntimeWarning: os.fork()
    # # was called.os.fork() is incompatible
    # # with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
    # # self.pid = os.fork()
    #
    # # 使用spawn而不是fork: Python的multiprocessing库默认使用fork来创建子进程
    # # 但在多线程应用中可能会出现问题。你可以通过设置multiprocessing的启动方式为spawn来避免这种问题。
    # multiprocessing.set_start_method("spawn")
    # 使用spawn太慢了
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called.")
    args = opts.parse_args()


    def _cal_metrics(tp, n, m):
        recall = float(tp) / m if m > 0 else 0
        precision = float(tp) / n if n > 0 else 0
        f1_score = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0

        print('Precision = ', round(precision, 4))
        print('Recall = ', round(recall, 4))
        print('F1-Score = ', round(f1_score, 4))

        return recall, precision, f1_score


    res_list = glob.glob(
        os.path.join(args.output, '*/best_res.csv')
    )
    # # 调试
    # print("best_res")
    # print(res_list)
    # 子目录
    params_list = glob.glob(
        os.path.join(args.output, '*/params_metrics.csv')
    )
    # # 调试
    # print("params_metrics")
    # print(params_list)

    df_list = []
    for res_file in res_list:
        df = pd.read_csv(res_file)
        df_list.append(df)

    # 参数量的计算
    df_params_list = []
    for params_file in params_list:
        df = pd.read_csv(params_file)
        df_params_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    full_df = list(full_df.sum(axis=0).values)

    # ValueError: No objects to concatenate
    # 为什么是空？需要表头吗？params_metrics.csv 和 best_res.csv  一样，操作也应该一样才对
    # # 调试
    # print(df_params_list)  # 检查是否为空
    full_params_df = pd.concat(df_params_list, ignore_index=True)
    full_params_df = list(full_params_df.sum(axis=0).values)

    micro_tp, micro_n, micro_m, macro_tp, \
        macro_n, macro_m, all_tp, all_n, all_m = full_df

    all_FLOPs, all_Params = full_params_df

    print(f'Micro result: TP:{micro_tp}, FP:{micro_n - micro_tp}, FN:{micro_m - micro_tp}')
    mic_rec, mic_pr, mic_f1 = _cal_metrics(micro_tp, micro_n, micro_m)

    print(f'Macro result: TP:{macro_tp}, FP:{macro_n - macro_tp}, FN:{macro_m - macro_tp}')
    mac_rec, mac_pr, mac_f1 = _cal_metrics(macro_tp, macro_n, macro_m)

    print(f'Total result: TP:{all_tp}, FP:{all_n - all_tp}, FN:{all_m - all_tp}')
    all_rec, all_pr, all_f1 = _cal_metrics(all_tp, all_n, all_m)

    print(f'Total FLOPs (GFLOPs): {all_FLOPs}')
    print(f'Total Params (M): {all_Params}')
