# import numpy as np
#
#
# def choose_neurons_to_monitor(net: Net, neurons_to_monitor_count: int):
#     neurons_to_monitor = {}
#     # n = {}
#     for klass in range(num_classes):
#         weightsStopSignClass = None
#         for name, param in net.named_parameters():
#             if name == "fc3.weight":
#                 # print(name, param.data[klass])
#                 weightsStopSignClass = param.data[klass].cpu().numpy()
#
#         absWeight = np.absolute(weightsStopSignClass)
#
#         neurons_to_monitor[klass] = absWeight.argsort()[::-1][:neurons_to_monitor_count]
#         # n[klass] = absWeight.argsort()[::-1][:neurons_to_monitor_count]
#
#     # print("neurons omitted for monitoring: " + str(len(neuronIndicesToBeOmitted[0])))
#     # print(f"neurons omitted for monitoring: {neuronIndicesToBeOmitted[0]}")
#     # print(f"neurons for monitoring: {n[0]}")
#
#     return neurons_to_monitor
#
#
# def choose_neurons_to_omit(net: Net, neurons_to_monitor_count: int):
#     neuronIndicesToBeOmitted = {}
#     # n = {}
#     for klass in range(num_classes):
#         weightsStopSignClass = None
#         for name, param in net.named_parameters():
#             if name == "fc3.weight":
#                 # print(name, param.data[klass])
#                 weightsStopSignClass = param.data[klass].cpu().numpy()
#
#         absWeight = np.absolute(weightsStopSignClass)
#
#         neuronIndicesToBeOmitted[klass] = absWeight.argsort()[::-1][neurons_to_monitor_count:]
#         print(f"omit: {neuronIndicesToBeOmitted[klass]}, {len(neuronIndicesToBeOmitted[klass])}")
#         print(
#             f"monitor: {absWeight.argsort()[::-1][:neurons_to_monitor_count]}, {len(absWeight.argsort()[::-1][:neurons_to_monitor_count])}")
#         exit(0)
#         # n[klass] = absWeight.argsort()[::-1][:neurons_to_monitor_count]
#
#     # print("neurons omitted for monitoring: " + str(len(neuronIndicesToBeOmitted[0])))
#     # print(f"neurons omitted for monitoring: {neuronIndicesToBeOmitted[0]}")
#     # print(f"neurons for monitoring: {n[0]}")
#
#     return neuronIndicesToBeOmitted
#
# def add_class_patterns_to_mymonitor(monitor: MyMonitor, loader: DataLoader, net: Net, device):
#     dataiter = iter(loader)
#     for img, label in dataiter:
#         label = label.to(device)
#         img = img.to(device)
#         # outputs, intermediate_values = net.forwardWithIntermediate(img)
#         # _, predicted = torch.max(outputs.data, 1)
#         predicted, intermediate_values = net.forwardWithIntermediate(img)
#         monitor.add_neuron_pattern(intermediate_values.cpu().numpy(), label.cpu().numpy())
#
# def write_csv(data, klass, filename, write_header=False):
#     fieldnames = ['class', 'comfort_level', 'correct']
#     # rows = [[klass, x[0], x[1]] for x in data]
#     # print(f"rows: {data}")
#     # for row in data:
#     #     print(row)
#     # print("write")
#     with open(filename, 'a', encoding='UTF8', newline='') as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(fieldnames)
#         writer.writerows(data)
#         f.flush()
#         os.fsync(f.fileno())
#
#
# import seaborn as sns
# import matplotlib.pyplot as plt
#
#
# def draw_boxplots():
#     # files = glob.glob("results*")
#     # for file in files:
#
#     df_cifar = pd.read_csv("results_cifarmy.csv")
#     df_test = pd.read_csv("results_testmy.csv")
#     incorrect = df_test[~df_test["correct"]]
#
#     percentage = df_test.groupby("comfort_level")["correct"].mean()
#     plt.figure()
#     _ = plt.plot(percentage)
#     plt.show()
#
#     concat = pd.concat([df_test, incorrect, df_cifar],
#                        keys=["test_set", "err_pred_test_set", "cifar"], names=["dataset_type", "num"])
#     concat.reset_index(inplace=True)
#     _ = sns.catplot(x="dataset_type", y="comfort_level", kind="box", data=concat)
#     plt.show()
#
#
# def find_threshold(filename_test, filename_cifar):
#     df_cifar = pd.read_csv(filename_cifar)
#     df_test = pd.read_csv(filename_test)
#     # comfort_level_count_cifar = df_cifar.groupby("comfort_level")["correct"].count().values
#     # comfort_level_count_test = df_test.groupby("comfort_level")["correct"].count().values
#     # print(comfort_level_count_cifar)
#     # print(comfort_level_count_test)
#     min = df_cifar["comfort_level"].min() if df_cifar["comfort_level"].min() > df_test["comfort_level"].min() else \
#     df_test["comfort_level"].min()
#     max = df_cifar["comfort_level"].max() if df_cifar["comfort_level"].max() > df_test["comfort_level"].max() else \
#         df_test["comfort_level"].max()
#     best_acc = 0
#     best = 0
#     for i in range(min - 1, max + 1):
#         curr = 0
#         # curr += sum(comfort_level_count_cifar[i + 1:])
#         # curr += sum(comfort_level_count_test[:i + 1])
#         curr += (df_cifar["comfort_level"] > i).sum()
#         curr += (df_test["comfort_level"] <= i).sum()
#         if best_acc < curr:
#             best_acc = curr
#             best = i
#     print(f" best threshold: {best}")
#     print(f" accuracy: {best_acc / (len(df_cifar.index) + len(df_test.index))}")
#     return best, best_acc / (len(df_cifar.index) + len(df_test.index))
#
#
# def find_best_neurons_count():
#     net = Net()
#     net.load_state_dict(torch.load('models/napmodel_5.pth'))
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     net.cuda(device)
#     neurons_to_monitor = choose_neurons_to_monitor(net, neurons_not_omitted_count)
#     monitor = MyMonitor(num_classes, neurons_to_monitor)
#
#     data = ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training', transform=standard_transform)
#
#     # from data import initialize_data, data_transforms, data_jitter_hue, data_jitter_brightness, data_jitter_saturation, \
#     #     data_jitter_contrast, data_rotate, data_hvflip, data_shear, data_translate, data_center, data_hflip
#
#     # loader = torch.utils.data.DataLoader(
#     #     torch.utils.data.ConcatDataset([ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_transforms),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_jitter_brightness),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_jitter_hue),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_jitter_contrast),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_jitter_saturation),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_translate),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_rotate),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_hvflip),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_center),
#     #                                     ImageFolder(root='data/GTSRB-Training_fixed/GTSRB/Training',
#     #                                                 transform=data_shear)]), batch_size=64, shuffle=True, num_workers=4,
#     #     pin_memory=True)
#
#     loader = DataLoader(data, batch_size=batch_size, shuffle=False)
#
#     cifar_testdata = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
#                                                   transform=standard_transform)
#
#     testdata = ImageFolder(root='data/GTSRB_Online-Test-Images-Sorted/GTSRB/Online-Test-sort',
#                            transform=standard_transform)
#     testloader = DataLoader(testdata, shuffle=False)
#     cifar_testloader = DataLoader(cifar_testdata, shuffle=False)
#
#     with torch.no_grad():
#         add_class_patterns_to_mymonitor(monitor, loader, net, device)
#         for i in range(149, neurons_to_monitor_count+1):
#             neurons_to_monitor = choose_neurons_to_monitor(net, i)
#             monitor.set_neurons_to_monitor(neurons_to_monitor)
#             filename_test = "results_testmy" + str(i) + ".csv"
#             filename_cifar = "results_cifarmy" + str(i) + ".csv"
#             process_dataset(filename_test, testloader, device, net, monitor)
#             process_dataset(filename_cifar, cifar_testloader, device, net, monitor)
#             print(f"n to monitor: {i}")
#             find_threshold(filename_test, filename_cifar)
#             import os
#             # os.remove(filename_test)
#             # os.remove(filename_cifar)
#
#
# def process_dataset(result_filename, testloader, device, net, monitor):
#     comfort_level_data = []
#     testiter = iter(testloader)
#     counter = 0
#
#     for imgs, label in tqdm(testiter):
#         label = label.to(device)
#         imgs = imgs.to(device)
#         outputs, intermediate_values = net.forwardWithIntermediate(imgs)
#         _, predicted = torch.max(outputs.data, 1)
#         # predicted, intermediate_values = net.forwardWithIntermediate(imgs)
#         correct_bitmap = (predicted == label)
#
#         for exampleIndex in range(intermediate_values.shape[0]):
#             lvl = monitor.get_comfort_level(intermediate_values.cpu().numpy()[exampleIndex, :],
#                                             predicted.cpu().numpy()[exampleIndex], omit=True)
#             comfort_level_data.append(
#                 (label.cpu().numpy()[exampleIndex], lvl,
#                  correct_bitmap.cpu().numpy()[exampleIndex]))
#             counter += 1
#             if counter == examples_count:
#                 write_csv(comfort_level_data, label.cpu().numpy()[exampleIndex], result_filename, write_header=True)
#                 return
