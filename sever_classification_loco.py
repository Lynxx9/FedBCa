import os
import argparse
import random
import copy
import torch.utils.data
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch import optim
from Statistical_method import *
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
from dataloader.Data_loader import DatasetGenerator
from train_cls import train_step_cls, train_step_cls_prox
from communication_method import communication
from models.ResNet_withoutBN import resnet50
from models.ResNet_Pre import Resnet2d, Resnet2d50


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-FL_mode', '--mode', type=str, default="FedAvg")
parser.add_argument('-dataset_n', '--dataset_name', type=str, default="BCa_d")
parser.add_argument('-class', '--num_class', type=int, default=2)
parser.add_argument('-rd', '--record_data', type=bool, default=True)
parser.add_argument('-g', '--gpu', type=str, default='0')
parser.add_argument('-E', '--epoch', type=int, default=1)
parser.add_argument('-B', '--batchsize', type=int, default=16)
parser.add_argument('-mn', '--model_name', type=str, default='Resnet2d')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.00001)
parser.add_argument('-vf', "--val_freq", type=int, default=5)
parser.add_argument('-sf', '--save_freq', type=int, default=100)
parser.add_argument('-ncomm', '--num_comm', type=int, default=200)
parser.add_argument('-sp', '--save_path', type=str, default='/home/yuihsin/FedBCa/results/')
parser.add_argument('--seed', type=int, default=57)
parser.add_argument('--lr_warm_epoch', type=int, default=20)
parser.add_argument('--lr_cos_epoch', type=int, default=580)
# leave_out_center=0 代表跑全部 4 輪 LOCO，1~4 代表只跑指定的那輪
parser.add_argument('--leave_out_center', type=int, default=0,
                    help='LOCO target center: 1~4 跑單輪, 0 跑全部四輪')


# ── 資料路徑 ──────────────────────────────────────────────────────────────────
CENTER_H5   = {
    1: "/home/yuihsin/FedBCa/process_classification/Center1/h5_data_128",
    2: "/home/yuihsin/FedBCa/process_classification/Center2/h5_data_128",
    3: "/home/yuihsin/FedBCa/process_classification/Center3/h5_data_128",
    4: "/home/yuihsin/FedBCa/process_classification/Center4/h5_data_128",
}
CENTER_TRAIN = {
    1: "/home/yuihsin/FedBCa/process_classification/Center1/center1_train.csv",
    2: "/home/yuihsin/FedBCa/process_classification/Center2/center2_train.csv",
    3: "/home/yuihsin/FedBCa/process_classification/Center3/center3_train.csv",
    4: "/home/yuihsin/FedBCa/process_classification/Center4/center4_train.csv",
}
CENTER_TEST = {
    1: "/home/yuihsin/FedBCa/process_classification/Center1/center1_test.csv",
    2: "/home/yuihsin/FedBCa/process_classification/Center2/center2_test.csv",
    3: "/home/yuihsin/FedBCa/process_classification/Center3/center3_test.csv",
    4: "/home/yuihsin/FedBCa/process_classification/Center4/center4_test.csv",
}


# ── Utility ───────────────────────────────────────────────────────────────────

def save_folder_mk(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        os.makedirs(os.path.join(path, "checkpoints"))
        os.makedirs(os.path.join(path, "prediction"))


def record_and_save(preds_list, labels_list, val_loss=None):
    preds_array  = np.array(preds_list)
    labels_array = np.squeeze(np.array(labels_list))
    AUC_value       = metrics.roc_auc_score(labels_array, preds_array)
    ACC_value       = get_accuracy(pred_value=preds_array,    label_value=labels_array)
    SEN_value       = get_sensitivity(pred_value=preds_array, label_value=labels_array)
    SPE_value       = get_specificity(pred_value=preds_array, label_value=labels_array)
    PREC_value      = get_precision(pred_value=preds_array,   label_value=labels_array)
    Threshold_value = get_best_threshold(pred_value=preds_array, label_value=labels_array)
    val_loss_str = f"  Val Loss:{val_loss:.4f}" if val_loss is not None else ""
    print(f"AUC:{AUC_value:.4f}  ACC:{ACC_value:.4f}  "
          f"SEN:{SEN_value:.4f}  SPE:{SPE_value:.4f}  PRE:{PREC_value:.4f}"
          + val_loss_str)
    return AUC_value, ACC_value, SEN_value, SPE_value, PREC_value, Threshold_value


# ── 單輪 LOCO ─────────────────────────────────────────────────────────────────

def run_one_loco(args, leave_out, base_save_path, dev):
    """
    單輪 LOCO：
      leave_out     : target center (1~4)，不參與訓練，只拿 test set 評估
      base_save_path: 上層結果目錄
      dev           : torch device
    回傳: best_auc (float)
    """
    source_ids     = [i for i in [1, 2, 3, 4] if i != leave_out]
    source_clients = [f"client{i}" for i in source_ids]
    target_client  = f"client{leave_out}"
    n_source       = len(source_clients)   # 固定 3

    print(f"\n{'='*60}")
    print(f"  FedAvg LOCO  |  Target: Center {leave_out}  |  Sources: {source_ids}")
    print(f"{'='*60}\n")

    # ── 固定隨機數（每輪 LOCO 用相同 seed，確保可重現）─────────────────
    seed = args['seed']
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    # ── Save path（每輪獨立子目錄）──────────────────────────────────────
    now       = datetime.datetime.now()
    save_path = os.path.join(base_save_path,
                             f"LOCO_target{leave_out}" + now.strftime("_%Y-%m-%d_%H.%M.%S"))
    save_folder_mk(save_path)

    # ── Log ──────────────────────────────────────────────────────────────
    test_txt = None
    if args['record_data']:
        log_path = os.path.join(save_path,
                                f"result_LOCO_target{leave_out}"
                                + now.strftime("_%Y-%m-%d_%H.%M.%S") + ".txt")
        test_txt = open(log_path, mode="a")
        test_txt.write(f"FedAvg LOCO — Target Center: {leave_out}\n")
        test_txt.write(f"Source centers: {source_ids}\n")
        test_txt.write("*"*10 + "parameter setting" + "*"*10 + "\n")
        test_txt.write(f"mode: {args['mode']}\n")
        test_txt.write(f"num_comm: {args['num_comm']}\n")
        test_txt.write(f"epoch: {args['epoch']}\n")
        test_txt.write(f"batchsize: {args['batchsize']}\n")
        test_txt.write(f"learning_rate: {args['learning_rate']}\n")
        test_txt.write(f"seed: {args['seed']}\n\n")

    # ── DataLoaders ───────────────────────────────────────────────────────
    Center_train_loader = {}
    Center_test_loader  = {}
    for cid in [1, 2, 3, 4]:
        train_ds = DatasetGenerator(path=CENTER_H5[cid], excelpath=CENTER_TRAIN[cid],
                                    Aug=True,  n_class=args['num_class'], set_name='train')
        test_ds  = DatasetGenerator(path=CENTER_H5[cid], excelpath=CENTER_TEST[cid],
                                    Aug=False, n_class=args['num_class'], set_name='test')
        Center_train_loader[f"client{cid}"] = torch.utils.data.DataLoader(
            train_ds, batch_size=args['batchsize'],
            shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
        Center_test_loader[f"client{cid}"]  = torch.utils.data.DataLoader(
            test_ds,  batch_size=args['batchsize'],
            shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # ── Model（每輪重新初始化，不跨輪共享權重）──────────────────────────
    net = Resnet2d50(in_channel=3, label_category_dict=dict(label=args['num_class']), dim=2)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func     = F.cross_entropy
    learning_rate = args['learning_rate']

    # ── 每個 source client 有各自獨立的 model 和 optimizer ──────────────
    models     = {c: copy.deepcopy(net) for c in source_clients}
    optimizers = {c: optim.Adam(models[c].parameters(), lr=learning_rate)
                  for c in source_clients}
    client_weights = [1.0 / n_source] * n_source   # FedAvg 等權重

    # ── Records ────────────────────────────────────────────────────────────
    sever_model       = net
    Test_all_AUC_list = []
    target_auc_record = np.array(["Epoch", "ACC", "SEN", "SPEC", "PREC", "Threshold", "AUC"])
    train_loss_list   = []
    train_lr_list     = []
    Max_AUC           = 0.0
    best_metrics      = {}
    val_loss_list     = []   # target center validation loss   # 紀錄 best AUC 那輪的完整指標

    # ══════════════════════════════════════════════════════════════════════
    # FL 訓練迴圈
    # ══════════════════════════════════════════════════════════════════════
    for i in range(args['num_comm']):
        print(f"\n--- Comm round {i+1} / {args['num_comm']} "
              f"| Target: Center{leave_out} ---")

        # ── Local training ────────────────────────────────────────────────
        for E in range(args['epoch']):
            for k, client in enumerate(tqdm(source_clients,
                                            desc=f"Round {i+1} local train")):
                if args['mode'].lower() == 'fedprox' and i > 0:
                    current_lr, loss_value = train_step_cls_prox(
                        train_loader=Center_train_loader[client],
                        model=models[client],
                        epoch=i,
                        optimizer=optimizers[client],
                        criterion=loss_func,
                        args=args,
                        sever_model=sever_model,
                    )
                else:
                    current_lr, loss_value, _ = train_step_cls(
                        train_loader=Center_train_loader[client],
                        model=models[client],
                        epoch=i,
                        optimizer=optimizers[client],
                        criterion=loss_func,
                        args=args,
                        global_parameters=sever_model.state_dict(),
                    )

        train_loss_list.append(loss_value)
        train_lr_list.append(current_lr)

        # ── FedAvg 聚合（只聚合 source clients）──────────────────────────
        sever_model, models = communication(
            args['mode'], sever_model, models, client_weights, source_clients)
        net = sever_model

        # ── 驗證：只測 target center 的 test set ──────────────────────────
        if (i + 1) % args['val_freq'] == 0:
            print(f"\n{'*'*10} Eval @ Comm {i+1} — Target: {target_client} {'*'*10}")

            Client_test_preds_list  = []
            Client_test_labels_list = []
            Client_index_slice_list = []
            val_loss_total = 0.0
            val_loss_steps = 0

            for num, (image, mask, label, sign) in enumerate(
                    Center_test_loader[target_client]):
                torch.set_grad_enabled(False)
                image  = image.to(dev)
                label_dev = label.to(dev)
                output = net(image)
                output = list(output.values())[0]

                # ── Validation Loss（slice-level cross-entropy）──────────
                val_loss = loss_func(output, np.squeeze(label_dev).long())
                val_loss_total += val_loss.item()
                val_loss_steps += 1

                for count, index_slice in enumerate(sign):
                    Client_test_preds_list.append(
                        np.array(output.cpu().detach().numpy()[:, 1])[count])
                    Client_test_labels_list.append(
                        np.squeeze(label.cpu().detach().numpy().astype(int))[count])
                    Client_index_slice_list.append(
                        int(index_slice.cpu().detach().numpy()))

            val_loss_avg = val_loss_total / val_loss_steps if val_loss_steps > 0 else 0.0
            val_loss_list.append(val_loss_avg)

            # Tumour-level aggregation
            Client_preds_mean_list  = []
            Client_labels_mean_list = []
            id_list_simple = list(set(Client_index_slice_list))

            for id_c in id_list_simple:
                index_same   = np.where(np.array(Client_index_slice_list) == id_c)
                tumour_pred  = np.mean(np.array(Client_test_preds_list)[index_same])
                tumour_label = int(round(np.mean(
                    np.array(Client_test_labels_list)[index_same])))
                Client_preds_mean_list.append(tumour_pred)
                Client_labels_mean_list.append(tumour_label)

            AUC_value, ACC_value, SEN_value, SPE_value, PREC_value, Threshold_value = \
                record_and_save(Client_preds_mean_list, Client_labels_mean_list, val_loss=val_loss_avg)

            Test_all_AUC_list.append(AUC_value)

            # 每次 val 都更新 last_metrics，迴圈結束時就是最後一輪的結果
            last_metrics = {
                "AUC":  AUC_value,
                "ACC":  ACC_value,
                "SEN":  SEN_value,
                "SPE":  SPE_value,
                "PREC": PREC_value,
                "Comm": i + 1,
            }

            target_auc_record = np.vstack((target_auc_record, np.array([
                i + 1,
                format(ACC_value,       ".3f"),
                format(SEN_value,       ".3f"),
                format(SPE_value,       ".3f"),
                format(PREC_value,      ".3f"),
                format(Threshold_value, ".3f"),
                format(AUC_value,       ".3f"),
            ])))

            # 儲存每輪預測值
            pred_df = pd.DataFrame({
                "id":    id_list_simple,
                "pred":  Client_preds_mean_list,
                "label": Client_labels_mean_list,
            })
            pred_df.to_excel(
                os.path.join(save_path, "prediction",
                             f"{target_client}_comm{i+1}_pred.xlsx"),
                index=False, float_format="%.5f")

            # Log
            if test_txt:
                test_txt.write(
                    f"\nComm {i+1} | train_loss={float(loss_value):.4f} | val_loss={val_loss_avg:.4f}\n"
                    f"  Target Center{leave_out}: "
                    f"AUC={AUC_value:.4f}  ACC={ACC_value:.4f}  "
                    f"SEN={SEN_value:.4f}  SPE={SPE_value:.4f}  "
                    f"PRE={PREC_value:.4f}  Thresh={Threshold_value:.4f}\n")

            # 儲存 Excel 紀錄
            excel_save_path = os.path.join(save_path, 'record.xlsx')
            record_writer   = pd.ExcelWriter(excel_save_path)
            pd.DataFrame(target_auc_record).to_excel(
                record_writer, f"target_center{leave_out}", float_format='%.5f')
            pd.DataFrame({
                "comm_round": [r[0] for r in target_auc_record[1:]],
                "AUC":        [r[-1] for r in target_auc_record[1:]],
            }).to_excel(record_writer, "auc_history", float_format='%.5f')
            record_writer.close()

            # 儲存最佳 model
            if AUC_value > Max_AUC:
                Max_AUC = AUC_value
                best_metrics = {
                    "AUC":  AUC_value,
                    "ACC":  ACC_value,
                    "SEN":  SEN_value,
                    "SPE":  SPE_value,
                    "PREC": PREC_value,
                    "Comm": i + 1,
                }
                torch.save(net, os.path.join(
                    save_path, "checkpoints",
                    f"comm{i+1}_AUC{Max_AUC:.4f}.pth"))
                print(f"  ★ New best AUC: {Max_AUC:.4f} @ Comm {i+1}")

        # 固定頻率存 checkpoint
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(
                save_path, "checkpoints", f"comm{i+1}.pth"))

    # ── 訓練結束，彙總此輪 ─────────────────────────────────────────────────
    best_auc        = max(Test_all_AUC_list)
    best_idx        = Test_all_AUC_list.index(best_auc)
    best_comm_round = (best_idx + 1) * args['val_freq']
    last_auc        = Test_all_AUC_list[-1]
    last_comm_round = len(Test_all_AUC_list) * args['val_freq']

    print(f"\n{'='*60}")
    print(f"  LOCO Target Center {leave_out}")
    print(f"  Best AUC : {best_auc:.4f} @ Comm ~{best_comm_round}")
    print(f"  Last AUC : {last_auc:.4f} @ Comm ~{last_comm_round}  ← 論文對齊")
    print(f"{'='*60}\n")

    if test_txt:
        test_txt.write(f"\n{'='*40}\n")
        test_txt.write(f"Best AUC : {best_auc:.4f} @ ~Comm {best_comm_round}\n")
        test_txt.write(f"Last AUC : {last_auc:.4f} @ ~Comm {last_comm_round} (論文對齊)\n")
        test_txt.close()

    # ── 畫圖 ───────────────────────────────────────────────────────────────
    val_rounds   = [(j + 1) * args['val_freq'] for j in range(len(Test_all_AUC_list))]
    train_rounds = list(range(1, len(train_loss_list) + 1))

    # ── 圖一：AUC curve（單獨）────────────────────────────────────────────
    plt.figure()
    plt.plot(val_rounds, Test_all_AUC_list, marker='o', markersize=3)
    plt.axhline(y=best_auc, color='r', linestyle='--',
                label=f'Best AUC={best_auc:.4f}')
    plt.title(f"FedAvg LOCO — Target Center {leave_out}")
    plt.xlabel("Communication Round")
    plt.ylabel("AUC (Target Center Test Set)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_path, f"AUC_curve_target{leave_out}.png"))
    plt.close()

    # ── 圖二：Train Loss + Val Loss + Target AUC 三線圖 ─────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # 左軸：Train Loss + Validation Loss
    color_train = '#2196F3'
    color_val   = '#FF9800'
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Loss", color='black')
    ax1.plot(train_rounds, train_loss_list,
             color=color_train, alpha=0.6, linewidth=1, label="Train Loss (Source)")
    ax1.plot(val_rounds, val_loss_list,
             color=color_val, marker='s', markersize=3,
             linewidth=1.5, label="Val Loss (Target)")
    ax1.tick_params(axis='y', labelcolor='black')

    # 右軸：Target AUC
    ax2 = ax1.twinx()
    color_auc = '#F44336'
    ax2.set_ylabel("AUC (Target Center Test Set)", color=color_auc)
    ax2.plot(val_rounds, Test_all_AUC_list,
             color=color_auc, marker='o', markersize=3,
             linewidth=1.5, label="Target AUC")
    ax2.axhline(y=best_auc, color=color_auc, linestyle='--', alpha=0.5,
                label=f'Best AUC={best_auc:.4f}')
    ax2.tick_params(axis='y', labelcolor=color_auc)
    ax2.set_ylim(0.4, 1.05)

    # 合併 legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

    plt.title(f"FedAvg LOCO — Target Center {leave_out}  "
              f"(Train Loss vs Val Loss vs Target AUC)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,
                             f"Loss_AUC_curve_target{leave_out}.png"), dpi=150)
    plt.close()

    # ── 圖三：Train Loss（單獨保留）──────────────────────────────────────
    plt.figure()
    plt.plot(train_rounds, train_loss_list, color='#2196F3', linewidth=1)
    plt.title("Train Loss Curve")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_path, "Loss_curve.png"))
    plt.close()

    # ── 圖四：LR curve ────────────────────────────────────────────────────
    plt.figure()
    plt.plot(range(len(train_lr_list)), train_lr_list)
    plt.title("Learning Rate Curve")
    plt.xlabel("Communication Round")
    plt.ylabel("LR")
    plt.savefig(os.path.join(save_path, "LR_curve.png"))
    plt.close()

    return best_metrics, last_metrics


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    args = parser.parse_args()
    args = args.__dict__
    args["IID"] = "center"

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # base_save_path: 所有 LOCO 輪次的上層目錄
    now            = datetime.datetime.now()
    base_save_path = os.path.join(
        args['save_path'],
        "FedAvg_LOCO_all" + now.strftime("_%Y-%m-%d_%H.%M.%S"))
    os.makedirs(base_save_path, exist_ok=True)

    # leave_out_center=0 → 跑全部四輪；否則只跑指定那輪
    if args['leave_out_center'] == 0:
        loco_targets = [1, 2, 3, 4]
    else:
        loco_targets = [args['leave_out_center']]

    # ── 逐輪跑 LOCO ────────────────────────────────────────────────────────
    best_results = {}   # {leave_out: best_metrics}
    last_results = {}   # {leave_out: last_metrics}（論文對齊）

    for leave_out in loco_targets:
        best_metrics, last_metrics = run_one_loco(args, leave_out, base_save_path, dev)
        best_results[leave_out] = best_metrics
        last_results[leave_out] = last_metrics

    # ── 最終彙總表格 ────────────────────────────────────────────────────────
    if len(best_results) > 0:
        metrics_keys = ["AUC", "ACC", "SEN", "SPE", "PREC"]
        col_w = 10

        def fmt_row(label, m_dict, show_comm=True):
            row = "  " + label.center(8)
            for k in metrics_keys:
                row += f"{m_dict[k]:.4f}".center(col_w)
            if show_comm:
                row += "  " + str(m_dict["Comm"]).rjust(10)
            return row

        def print_table(title, results):
            avg = {k: np.mean([results[c][k] for c in results]) for k in metrics_keys}
            header = ("  " + "Target".center(8)
                      + "".join(k.center(col_w) for k in metrics_keys)
                      + "  " + "Comm".rjust(10))
            sep = "  " + "-" * (8 + col_w * len(metrics_keys) + 12)
            print("\n" + "=" * 62)
            print(f"  {title}")
            print("=" * 62)
            print(header)
            print(sep)
            for c in sorted(results.keys()):
                print(fmt_row(f"Center {c}", results[c], show_comm=True))
            print(sep)
            print(fmt_row("Average", avg, show_comm=False))
            print("=" * 62)
            return avg, header, sep

        # 印兩張表
        best_avg, best_header, best_sep = print_table(
            "Best AUC（500輪中最高點）", best_results)
        last_avg, last_header, last_sep = print_table(
            "Last AUC（最後一輪，與論文對齊）★", last_results)

        # ── 儲存彙總 txt ──────────────────────────────────────────────────
        summary_path = os.path.join(base_save_path, "LOCO_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write("FedAvg LOCO Summary\n")
            for title, results, avg, header, sep in [
                ("Best AUC（500輪中最高點）", best_results, best_avg, best_header, best_sep),
                ("Last AUC（最後一輪，與論文對齊）★", last_results, last_avg, last_header, last_sep),
            ]:
                sf.write("\n" + "=" * 62 + "\n")
                sf.write(f"  {title}\n")
                sf.write("=" * 62 + "\n")
                sf.write(header + "\n" + sep + "\n")
                for c in sorted(results.keys()):
                    sf.write(fmt_row(f"Center {c}", results[c], show_comm=True) + "\n")
                sf.write(sep + "\n")
                sf.write(fmt_row("Average", avg, show_comm=False) + "\n")

        # ── 儲存彙總 Excel（兩個 sheet）──────────────────────────────────
        excel_path = os.path.join(base_save_path, "LOCO_summary.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            for sheet_name, results, avg in [
                ("Best_AUC", best_results, best_avg),
                ("Last_AUC_論文對齊", last_results, last_avg),
            ]:
                rows = []
                for c in sorted(results.keys()):
                    m = results[c]
                    rows.append({"Target Center": f"Center {c}",
                                 **{k: round(m[k], 4) for k in metrics_keys},
                                 "Comm": m["Comm"]})
                rows.append({"Target Center": "Average",
                             **{k: round(avg[k], 4) for k in metrics_keys},
                             "Comm": "-"})
                pd.DataFrame(rows).to_excel(writer, sheet_name=sheet_name,
                                            index=False, float_format="%.4f")

        print(f"\n彙總結果已存至: {base_save_path}")
