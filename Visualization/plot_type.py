import matplotlib.pyplot as plt
import math
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from Visualization.tensorboard import Tensorboard_elem, Tensorboard_node


def using_tensorboard(fileTensor, config, path, file_name):
    print('Using tensorboard')
    epoch_rows = math.ceil(60000.0/float(config['batch_size']))
    config['epoch_rows'] = epoch_rows
    if file_name is None:
        file_name = 'grad'

    # if config['nn_type'] == 'lenet5':
    if False:
        logger = Tensorboard_elem(fileTensor, path, file_name, config)
        if config['visual_type'] == 'time_domain':
            logger.time_write()
        elif config['visual_type'] == 'node_domain':
            logger.node_write()
    else:
        logger = Tensorboard_node(fileTensor, path, file_name, config)
        if config['visual_type'] == 'time_domain':
            logger.time_write()
        elif config['visual_type'] == 'node_domain':
            logger.node_write()

    # if config['nn_type']=='lenet5':
    #     from NeuralNet.lenet5 import w_size_list,b_size_list,NN_size_list,NN_type_list,kernel_size_list
    # if False:
    #     elemWriter=SummaryWriter(log_dir='visualizing_data/elem_info',)
    # else:
    #     nodeWriter=SummaryWriter(log_dir='visualizing_data/node_info/{}'.format(file_name))
    #     total_data_list=list()
    #     for t,line in enumerate(fileTensor):
    #         total_data_list.append(torch.tensor(line).clone().detach())

    #         if t%1000==0:
    #             print('\r {} line complete'.format(t),end='')
    #         for l,(num_w,num_b) in enumerate(zip(w_size_list,b_size_list)):# 라인 내의 모든 param처리
    #             tmp_w=torch.tensor(line[:num_w],dtype=torch.float).clone().detach()#layer 단위
    #             line=line[num_w:]
    #             nodeWriter.add_scalar('norm_grad/{}l'.format(l),tmp_w.norm(2),t)#norm
    #             if NN_type_list[l]=='cnn':
    #                 n=0
    #                 for n in range(NN_size_list[l+1]):#node 단위
    #                     node_w=tmp_w[:(kernel_size_list[l][0]*kernel_size_list[l][1])*NN_size_list[l]]
    #                     # print(tmp_w.size(),node_w.size(),NN_type_list[l],n," ",l," ",t)
    #                     nodeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
    #                     nodeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
    #                     tmp_w=tmp_w[(kernel_size_list[l][0]*kernel_size_list[l][1])*NN_size_list[l]:]# 내용 제거

    #             elif NN_type_list[l]=='fc':
    #                 n=0
    #                 for n in range(NN_size_list[l+1]):#node 단위
    #                     node_w=tmp_w[:NN_size_list[l]]
    #                     nodeWriter.add_scalar('sum_grad/{}l_{}n'.format(l,n),node_w.sum(),t)#합
    #                     nodeWriter.add_scalar('norm_grad/{}l_{}n'.format(l,n),node_w.norm(2),t)#norm
    #                     tmp_w= tmp_w[NN_size_list[l]:] # 내용제거

    #             tmp_b=torch.tensor(line[:num_b],dtype=torch.float).clone().detach()
    #             line=line[num_b:]#내용제거
    #             nodeWriter.flush()

    #     # 시간과 상관없는 분석을 위한 cat
    #     total_data=torch.cat(total_data_list,dim=0)
    #     nodeWriter.close()
    print('\n ==Visualization Complete==')


def using_plt(fileTensor, config, path):
    print('Using plt')
    NUM_ROWS = config['epochs']*math.ceil(60000.0/float(config['batch_size']))
    if config['nn_type'] == 'lenet5':
        from NeuralNet.lenet5 import w_size_list, b_size_list, NN_size_list, NN_type_list, kernel_size_list
        print(config['nn_type'])

    grad_data = list()
    weight_data = list()
    # sum of grad in all node 구조 만들기 (node에서의 elem들의 sum)
    sum_grad_w_node_list = [[[[]for _ in range(NUM_ROWS)] for _ in range(
        NN_size_list[i+1])] for i, w_size in enumerate(w_size_list)]
    # avg of grad in all node 구조 만들기 (node에서의 elem들의 평균 grad)
    avg_grad_w_node_list = [[[[]for _ in range(NUM_ROWS)] for _ in range(
        NN_size_list[i+1])] for i, w_size in enumerate(w_size_list)]
    dist_grad_w_node_list = [[[[]for _ in range(NUM_ROWS)] for _ in range(
        NN_size_list[i+1])] for i, w_size in enumerate(w_size_list)]

    for t, line in enumerate(fileTensor):
        line_float = list(map(float, line))
        grad_data.append(list())
        weight_data.append(list())
        if t % 1000 == 0:
            print('\r {} line complete'.format(t), end='')
        for l, (num_w, num_b) in enumerate(zip(w_size_list, b_size_list)):
            tmp_w = torch.tensor(line_float[:num_w])
            line_float = line_float[num_w:]
            grad_data[-1].append(tmp_w)
            weight_data[-1].append(tmp_w)

            if NN_type_list[l] == 'cnn':
                for n in range(NN_size_list[l+1]):
                    sum_grad = torch.tensor(tmp_w[:(
                        kernel_size_list[l][0]*kernel_size_list[l][1])*NN_size_list[l]]).clone().detach().sum().item()
                    dist_grad_w_node_list[l][n] = tmp_w[:(
                        kernel_size_list[l][0]*kernel_size_list[l][1])*NN_size_list[l]]
                    print('\r {} {} {}'.format(l, n, t), end='')
                    sum_grad_w_node_list[l][n][t] = sum_grad
                    avg_grad_w_node_list[l][n][t] = sum_grad / \
                        float(NN_size_list[l+1])
                    tmp_w = tmp_w[(kernel_size_list[l][0] *
                                   kernel_size_list[l][1]**2)*NN_size_list[l]:]

            elif NN_type_list[l] == 'fc':
                for n in range(NN_size_list[l+1]):
                    sum_grad = torch.tensor(
                        tmp_w[:NN_size_list[l]]).sum().clone().detach().item()
                    dist_grad_w_node_list[l][n] = tmp_w[:NN_size_list[l]]
                    sum_grad_w_node_list[l][n][t] = sum_grad
                    avg_grad_w_node_list[l][n][t] = sum_grad / \
                        float(NN_size_list[l+1])
                    tmp_w = tmp_w[NN_size_list[l]:]

            tmp_b = torch.tensor(line_float[:num_b])
            line_float = line_float[num_b:]
            grad_data[-1].append(tmp_b)

    # 1 Box plot for checking distribution
    box_w = [list() for i in weight_data[0]]
    for i, node_w in enumerate(weight_data):
        for j, elem_w in enumerate(node_w):
            box_w[j].append(elem_w.tolist())
    print("No.1 Start")
    for j, _ in enumerate(w_size_list):
        plt.clf()
        plt.figure(figsize=(40, 5))
        plt.boxplot(box_w[j], labels=['{}'.format(k) for k, _ in enumerate(
            weight_data)], showmeans=True, autorange=True, whis=2)  # median 도 표시를 해주자
        plt.title('layer_grad_weight_distribution')
        plt.axis([-0.5, 50.5, -0.3, 0.3])
        plt.xlabel('epoch')
        plt.savefig(os.path.join(
            path, 'layer_grad_distribution/layer_grad_dist_{}.png'.format(j)), dpi=200)

    # 모든 element에 대해서 time기반 비교
    time_list = list()
    for t, _ in enumerate(weight_data):
        time_list.append(t)

    # 시간에 대해서 저장
    # layer elem time 순 -> elem_w_list
    elem_w_list = [[list() for _ in w] for w in weight_data[0]]  # 뒤에 시간별로 더하면됨

    for t, w in enumerate(weight_data):  # 모든 시간 t에 대해서
        for j, w_layer in enumerate(w):  # 각 layer에 대해서
            for i, elem_w in enumerate(w_layer):  # layer내에 대해서
                elem_w_list[j][i].append(elem_w)
    '''
    element 수준에서는 관찰하지 않음
    print("No.2 Start")
    #2 elem_w_list의 size: layer,weight_i
    # plot and save
    for j,w_node in enumerate(elem_w_list):
        for i,w_elem in enumerate(w_node):
            plt.clf()#clear figure
            plt.plot(time_list,w_elem)
            plt.title('layer{}_elem{}'.format(j,i))
            plt.xlabel('time(epoch)')
            plt.ylabel('sum_of_grad in all elem')
            plt.savefig(os.path.join(path,'layer_node_individ/layer{}_elem{}.png'.format(j,i)),dpi=100,facecolor='#eeeeee')
        print(j,'layer done')
    '''

    # time 기반 layer별로 gradient 값을 모두 합쳐서 비교
    # sum_w
    # layer,time,weight_sum(elem들의 float)
    sum_w = [[list() for i in range(NUM_ROWS)]
             for w in elem_w_list]  # layer내의/ 시간에 대한/ grad 요소들의 합
    for i, layer_w in enumerate(elem_w_list):  # layer구분
        for elem_w in layer_w:  # time값을 갖는 elem
            for t, elem_w in enumerate(elem_w):  # 각 time 의 elem
                sum_w[i][t] = 0.0
    for i, layer_w in enumerate(elem_w_list):  # layer구분
        for elem_w in layer_w:  # time값을 갖는 elem
            for t, elem in enumerate(elem_w):  # 각 time 의 elem
                sum_w[i][t] += elem.item()/float(w_size_list[i])  # averaging

    color_list = ['red', 'yellow', 'green', 'blue', 'black']
    print("No.3 Start")
    plt.clf()
    plt.figure()
    for j, (w, color) in enumerate(zip(sum_w, color_list)):
        plt.plot(time_list, w, color=color)  # log or not
    plt.title('all_layer_all_node'.format(j))
    plt.xlabel('time(epoch)')
    plt.ylabel('average of grad(by num of elem)')
    plt.legend(['l0', 'l1', 'l2', 'l3', 'l4'])
    plt.savefig(os.path.join(path, 'all_node_all_layer.png'),
                dpi=200, facecolor='#eeeeee')

    # 모든 변화량 (x축), element of weight
    # y축은 모든 시간에 대한 gradient의 합
    # layer별로 다르게 분할
    sum_grad_in_time = [[list() for i in w] for w in elem_w_list]
    sum_grad_in_layer_n_time = list()
    for i, layer_w in enumerate(elem_w_list):  # layer구분
        for j, elem_w in enumerate(layer_w):  # time값을 갖는 elem
            sum_grad = torch.tensor(elem_w).sum().item()  # 모든 시간에 대한 합
            sum_grad_in_time[i][j] = sum_grad
        sum_grad_in_layer_n_time.append(
            torch.tensor(sum_grad_in_time[i]).mean())

    print("No.4 Start")
    # 4 모든 layer에 대해서 각 elem(x) 시간에 따른 변화량의 합(y)
    num_node_list = [len(w) for w in elem_w_list]
    for i, (num_node, sum_grad_time) in enumerate(zip(num_node_list, sum_grad_in_time)):
        x_axis = [i for i in range(num_node)]
        plt.clf()  # clear figure
        plt.figure()
        plt.scatter(x_axis, sum_grad_time)
        plt.title('{}_layer_all_node'.format(i))
        plt.xlabel('element_num')
        plt.ylabel('sum of grad in all the times')
        plt.savefig(os.path.join(path, '{}_layer_all_node.png'.format(
            i)), dpi=200, facecolor='#eeeeee')
    print("No.5 Start")
    # 5 모든 layer(x)에서 (모든 시간과 (elem의 평균))(y)
    x_axis = [i for i, _ in enumerate(w_size_list)]
    plt.clf()  # clear figure
    plt.figure()
    plt.scatter(x_axis, sum_grad_in_layer_n_time)
    plt.title('sum of all elem in each layer'.format(i))
    plt.xlabel('layer')
    plt.ylabel('sum of gradient of all elems in all the times')
    plt.savefig(os.path.join(path, 'each_layer_all_node.png'.format(
        i)), dpi=200, facecolor='#eeeeee')

    print("No.6 Start")
    # 6 Node별 시간당 변화량
    for i, layer_w in enumerate(sum_grad_w_node_list):
        for j, node_w in enumerate(layer_w):
            plt.clf()  # clear figure
            plt.figure()
            plt.plot(time_list, node_w)
            plt.xlabel('time(epoch)')
            plt.ylabel('sum of grad in node')
            plt.title('sum of grad in layer{} node{}'.format(i, j))
            plt.savefig(os.path.join(path, 'node_info', 'sum_of_grad_{}layer_{}node.png'.format(
                i, j)), dpi=200, facecolor='#eeeeee')

    # 6-1 Node별 시간당 변화량 (겹치기)
    for i, layer_w in enumerate(sum_grad_w_node_list):
        plt.clf()  # clear figure
        plt.figure(figsize=(7, 10))
        legend_list = list()
        for j, node_w in enumerate(layer_w):
            legend_list.append('node'+str(j))
            plt.plot(time_list, node_w)
        plt.legend(legend_list)
        plt.xlabel('time(epoch)')
        plt.ylabel('sum of grad in node')
        plt.title('sum of grad in layer{}'.format(i))
        plt.savefig(os.path.join(path, 'node_info', 'sum_of_grad_{}layer.png'.format(
            i)), dpi=200, facecolor='#eeeeee')

    # 6-2 expectation of all elems (node:x, time:y)
    for i, layer_w in enumerate(avg_grad_w_node_list):
        for j, node_w in enumerate(layer_w):
            plt.clf()  # clear figure
            plt.figure()
            plt.plot(time_list, node_w, color='red')
            plt.xlabel('time(epoch)')
            plt.ylabel('avg of grad in node')
            plt.title('avg of grad(all elems) in layer{} node{}'.format(i, j))
            plt.savefig(os.path.join(path, 'node_info', 'avg_of_grad_{}layer_{}node.png'.format(
                i, j)), dpi=200, facecolor='#eeeeee')

    # 6-3 expectation of all elems (node:x, time:y)(겹치기)
    for i, layer_w in enumerate(avg_grad_w_node_list):
        print(len(layer_w))
        plt.clf()  # clear figure
        plt.figure(figsize=(7, 10))
        legend_list = list()
        for j, node_w in enumerate(layer_w):
            legend_list.append('node'+str(j))
            plt.plot(time_list, node_w)
        plt.xlabel('time(epoch)')
        plt.legend(legend_list)
        plt.ylabel('avg of grad in node')
        plt.title('avg of grad in layer{}'.format(i))
        plt.savefig(os.path.join(path, 'node_info', 'avg_of_grad_{}layer.png'.format(
            i)), dpi=200, facecolor='#eeeeee')
