 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 10:09:48 2021

@author: zzh
"""
import csv
import numpy as np
import pandas as pd
class node:
    def __init__(self,ID):
        self.ID = ID
        self.edge_weigt = {}
        self.edge_id = []
        self.k = 0
        self.catagory = -1
        self.origin_ID = set()
    def add_edge(self,out_id,weight):
        if out_id in self.edge_weigt.keys():
            self.edge_weigt[out_id]+=weight
        else:
            self.edge_weigt[out_id] = weight
        self.k += weight
class graph:
    def __init__(self):
        self.m =0
        self.node_dict = {}
        self.edges_weight = {}
        self.total_cata = 0
        self.partion = {}
        self.partion_sum_total = {}
    def add_node(self,node):
        self.node_dict[node.ID] = node
    def add_edge_graph(self,in_id, out_id,weight):
        node_1 = self.node_dict[in_id]
        node_2 = self.node_dict[out_id]
        if (in_id,out_id) in self.edges_weight.keys():
            self.edges_weight[(in_id,out_id)]+=weight
        else:
            self.edges_weight[(in_id,out_id)] = weight
        node_1.add_edge(out_id,weight)
        node_2.add_edge(in_id,weight)
        self.m +=2*weight
    def initial_partion (self):
        i = 0 
        for keys,node in self.node_dict.items():
            self.partion[i] = [keys]
            self.partion_sum_total[i]=0
            self.partion_sum_total[i]+=node.k
            node.catagory = i
            i+=1
        self.total_cata = i
    '''
    def restructuring(self):
        size = len(self.partion.keys())
        new_edges = np.zeros(size,size)
        for edge in self.edges:
            node_1 = self.node_dict[edge[0]]
            node_2 = self.node_dict[edge[1]]  
            cata_1 = node_1.catagory
            cata_2 = node_2.catagory
            new_edges[cata_1][cata_2] +=1
            new_edges[cata_2][cata_1] +=1
        for i,node_ids in self.partion:
            super_node = node(i)
            super_node.catagory=i
            super_node.k = sum(new_edges[i,:])
            for node in node_ids:
      '''          
                
            
        
        
        
        
        
csvFile  = open('data\edges_update.csv')
reader = csv.reader(csvFile)
com_graph = graph()
for item in reader:
    if item[0] == 'Source':
        continue
    item[0] = int(item[0])
    item[1] = int(item[1])
    if item[0] not in com_graph.node_dict.keys():
        node_1 = node(item[0])
        node_1.origin_ID.add(item[0])
        com_graph.add_node(node_1)
    else:
        node_1 = com_graph.node_dict[item[0]]
    if item[1] not in com_graph.node_dict.keys():
        node_2 = node(item[1])
        node_2.origin_ID.add(item[1])
        com_graph.add_node(node_2)
    else:
        node_2 = com_graph.node_dict[item[1]]
#    node_1.add_edge(item[1],1)
#    node_2.add_edge(item[0],1)
    com_graph.add_edge_graph(item[0],item[1],1)
com_graph.initial_partion()
print('Finish building graph')
def phase_1(graph,flag1):
    #遍历每一个点
    list_temp = sorted(graph.node_dict.items(),key = lambda x:x[1].k,reverse = False)
    for ID, node in list_temp:
        max_cata = node.catagory
        max_gain = 0
        k_i = node.k
        k_i_in_2 = 0
        cata_list = []
        sum_total_2 = graph.partion_sum_total[node.catagory]-node.k
        list_temp1 = sorted(node.edge_weigt.items(),key = lambda x:graph.node_dict[x[0]].k,reverse =False )
        for id_nei, weight_nei in list_temp1:
            if id_nei == ID:
                continue
            if graph.node_dict[id_nei].catagory == node.catagory:
                k_i_in_2+=weight_nei
            else:
                continue
        #遍历该点的邻居
        '''
        for ids in graph.partion[node.catagory]:
            if ids == ID:
                continue
            else:
                sum_total_2 += graph.node_dict[ids].k
                '''
        '''
        for id_neigh,weight in node.edge_weigt.items():
            if id_neigh == ID:
                continue
            elif graph.node_dict[id_neigh].catagory != node.catagory:
                temp_decrease+=weight
            else :
                k_i_in_2+=weight
                temp_increase+=weight
        sum_total_2 = graph.partion_sum_total[node.catagory]-temp_decrease+temp_increase
        '''
        #遍历该点的邻居点
        list_temp2 = sorted(node.edge_weigt.items(),key = lambda x:graph.node_dict[x[0]].k,reverse = True)
        for id_neigh,weight in list_temp2:
            node_neigh = graph.node_dict[id_neigh]
            sum_total = 0
            k_i_in = 0
            gain = 0
            #记录该邻居的类别
            cata = node_neigh.catagory
            
            if cata == node.catagory:
                continue
            
            if cata in cata_list:
                continue
            else:
                cata_list.append(cata)
            sum_total = graph.partion_sum_total[cata]
            '''
            for ids in graph.partion[node_neigh.catagory]: 
                sum_total+=graph.node_dict[ids].k
            '''
            #遍历该点的邻居
            list_temp3 = sorted(node.edge_weigt.items(),key = lambda x:graph.node_dict[x[0]].k,reverse = False)
            for id_node,weight_node in list_temp3 :
                if id_node == ID:
                    continue
                if graph.node_dict[id_node].catagory == cata:
                    k_i_in += weight_node  
#                    temp_decrease += weight_node
                else:
                    continue
#                    temp_increase += weight_node
            #sum_total = sum_total+temp_increase-temp_decrease 
            gain = k_i_in/(2*graph.m)-0.5*(sum_total*k_i)/(graph.m*graph.m)
            delta_q2 = 0.5*(sum_total_2*k_i)/(graph.m*graph.m)-0.5*k_i_in_2/graph.m
            gain = gain+delta_q2 
            if gain > max_gain:
                max_gain = gain
                max_cata = cata
        if max_cata == node.catagory:
            continue
        else:
            #更改划分
            #将该点从原来的划分中删除
            index_cata_old = graph.partion[node.catagory].index(ID)
            graph.partion[node.catagory].pop(index_cata_old)
            #如果该类中的点的数量为0则删除该类
            if (len(graph.partion[node.catagory])==0):
                graph.total_cata-=1
                graph.partion.pop(node.catagory)
                graph.partion_sum_total.pop(node.catagory)
                #维护邻居类的sum_out
                graph.partion_sum_total[max_cata]+=node.k
                '''
                for nei_id, weight_nei in node.edge_weigt.items():
                    if nei_id in graph.partion[max_cata]:
                        graph.partion_sum_in[max_cata] += weight_nei
                    else:
                        if nei_id == node.ID:
                            graph.partion_sum_in[max_cata] += weight_nei
                        else:
                            graph.partion_sum_total[max_cata] += weight_nei
                '''
                
                       
            else:
                #维护邻居类的sum_out
                graph.partion_sum_total[node.catagory]-=node.k
                graph.partion_sum_total[max_cata]+=node.k
                '''
                for nei_id, weight_nei in node.edge_weigt.items():
                    #自环的情况
                    if nei_id == node.ID:
                        graph.partion_sum_in[node.catagory]-=weight_nei
                        graph.partion_sum_in[max_cata] += weight_nei
                    elif nei_id in graph.partion[node.catagory]:
                        graph.partion_sum_in[node.catagory]-=weight_nei
                        graph.partion_sum_total[node.catagory] += weight_nei
                        graph.partion_sum_total[max_cata] += weight_nei
                    else:
                        if nei_id in graph.partion[max_cata]:
                            graph.partion_sum_in[max_cata] += weight_nei
                            graph.partion_sum_total[max_cata] -= weight_nei
                            graph.partion_sum_total[node.catagory] -= weight_nei
                        else:
                            graph.partion_sum_total[node.catagory] -= weight_nei
                            graph.partion_sum_total[max_cata] += weight_nei
                '''
                
            graph.partion[max_cata].append(ID)
            node.catagory = max_cata
            flag1 = 1
    return graph,flag1
def phase_2(graph_old):
    new_graph = graph()
    #初始化新的点
    for new_node_id in graph_old.partion.keys():
        node_new = node(new_node_id)
        new_graph.add_node(node_new)
        new_graph.node_dict[new_node_id].catagory = new_node_id
        new_graph.partion[new_node_id] = [new_node_id]
        new_graph.partion_sum_total[new_node_id] = 0
        new_graph.total_cata+=1
    #初始化边
    for new_node_id, old_node_id_list in graph_old.partion.items():
        new_node = new_graph.node_dict[new_node_id]
        for old_id in old_node_id_list:
            old_node = graph_old.node_dict[old_id]
            new_node.origin_ID =new_node.origin_ID | old_node.origin_ID
            for old_nei_id,weight in old_node.edge_weigt.items():
                node_nei = graph_old.node_dict[old_nei_id]
                new_graph.add_edge_graph(old_node.catagory, node_nei.catagory, weight)
#                new_node.add_edge(node_nei.catagory,weight)

                if old_node.catagory == node_nei.catagory:
                    new_graph.partion_sum_total[old_node.catagory] += 2*weight
                else:
                    new_graph.partion_sum_total[old_node.catagory] += weight
                '''
                if node_nei.catagory == old_node.catagory:
                    new_graph.partion_sum_in[old_node.catagory] += weight
                    new_node.add_edge(node_nei.catagory,weight)
                else:
                    new_graph.partion_sum_total[old_node.catagory] += weight
                    new_node.add_edge(node_nei.catagory,weight)
                '''
                
    return new_graph
                    
        
def vote(graph):
 
    list_number = sorted(graph.node_dict.items(),key = lambda x:len(x[1].origin_ID),reverse = True)
    list_id_5 = []
    print(list_number[:5])
    ID_dict = {}
    dic_origin = {}
    final_dict ={}
    for i in range(len(list_number)):
        if i < 5 :
            index = list_number[i][0]
            ID_dict[index] = list_number[i][1].origin_ID
            dic_origin[i]=list_number[i][1].origin_ID
            list_id_5.append(index)
        else:
            dic_origin[i]=list_number[i][1].origin_ID
            node = graph.node_dict[list_number[i][0]]
            max_edge = -1
            max_id = -1
            for ids,nei in node.edge_weigt.items():
                if ids in list_id_5:
                    if nei>max_edge:
                        max_edge = nei
                        max_id = ids
                else:
                    continue
            
            ID_dict[max_id]=ID_dict[max_id]|node.origin_ID
    temp = 0
    for ids,origi_id in ID_dict.items():
        final_dict[temp] = origi_id
        temp+=1
    for ids,origi_id in dic_origin.items():
        print(len(origi_id))
    return final_dict,dic_origin

def ACC_compute(final_dict,dic_origin):
    csvFile  = open('ground_truth.csv')
    reader = csv.reader(csvFile)
    list_0=np.zeros(5)
    list_1=np.zeros(5)
    list_2=np.zeros(5)
    list_3=np.zeros(5)
    list_4=np.zeros(5)
    matrix_show = np.zeros((5,len(dic_origin.keys())))
    for item in reader:
        if item[1] == 'category':
            continue
        if int(item[1]) ==0:
            flag = 0
            for i in range(5):
                if int(item[0]) in final_dict[i]:
                    list_0[i]+=1
            for i in range(len(dic_origin.keys())):
                if int(item[0]) in dic_origin[i]:
                     matrix_show[0][i]+=1
        if int(item[1]) ==1:
            for i in range(5):
                if int(item[0]) in final_dict[i]:
                    list_1[i]+=1
            for i in range(len(dic_origin.keys())):
                if int(item[0]) in dic_origin[i]:
                     matrix_show[1][i]+=1
        if int(item[1]) ==2:
            for i in range(5):
                if int(item[0]) in final_dict[i]:
                    list_2[i]+=1
            for i in range(len(dic_origin.keys())):
                if int(item[0]) in dic_origin[i]:
                     matrix_show[2][i]+=1
        if int(item[1]) ==3:
            for i in range(5):
                if int(item[0]) in final_dict[i]:
                    list_3[i]+=1
            for i in range(len(dic_origin.keys())):
                if int(item[0]) in dic_origin[i]:
                     matrix_show[3][i]+=1
        if int(item[1]) ==4:
            for i in range(5):
                if int(item[0]) in final_dict[i]:
                    list_4[i]+=1
            for i in range(len(dic_origin.keys())):
                if int(item[0]) in dic_origin[i]:
                     matrix_show[4][i]+=1
    print(list_0)
    print(list_1)
    print(list_2)
    print(list_3)
    print(list_4)
    print('*'*20)
    print(matrix_show)
    dic_clus_map ={}
    map_vector = np.argmax(matrix_show,axis=0)
    for i in range(len(dic_origin.keys())):
        dic_clus_map[i] = map_vector[i]
    
    total = 0
    correct = 0
    csvFile  = open('ground_truth.csv')
    reader = csv.reader(csvFile)
    for item in reader:
        predi = -1
        if item[1] == 'category':
            continue
        else:
            for i in range(len(dic_origin.keys())):
                if int(item[0]) in dic_origin[i]:
                    predi = dic_clus_map[i]
        if predi == int(item[1]):
            correct+=1
        total+=1
    return correct/total, dic_clus_map
            
            
flag1=1
t=0
while(1):
    node_number = len(com_graph.node_dict.keys())
    print('点数量')
    print(node_number)
    if node_number == 5:
        break;
    i = 0
    while(flag1):
        flag1=0
        node_number = len(com_graph.node_dict.keys())
        com_graph ,flag1 = phase_1(com_graph,flag1)
        print("num of lei")
        print(com_graph.total_cata)
        if com_graph.total_cata == 5:
            break
        i+=1
    if i == 1:
        break
    print('phase_2')
    com_graph = phase_2(com_graph)
    print(com_graph.total_cata)
    flag1 = 1

final_dict,dic_origin = vote(com_graph)
acc,dic_clus_map = ACC_compute(final_dict,dic_origin)
print(acc)
csvFile  = open('sample_res.csv.csv')
reader = csv.reader(csvFile)
list_id = []
cata_list = []
for item in reader:
    if item[0] == 'id':
        continue
    ids = int(item[0])
    list_id.append(ids)
    for i in range(len(dic_origin.keys())):
                if int(item[0]) in dic_origin[i]:
                    predi = dic_clus_map[i] 
    cata_list.append(predi)

frame = pd.DataFrame({'id':list_id,'category':cata_list})
frame.to_csv('res.csv',index=False)


'''
flag2 = 1
m = com_graph.m
while(flag1):
    while(flag2):
        flag2 = 0
        for ID,node in com_graph.node_dict.items():
            max_cata = node.catagory
            max_gain = 0
            k_i = node.k
            for id_neigh in node.edge_id:
                node_neigh = com_graph.node_dict[id_neigh]
                sum_in = 0
                sum_total = 0
                k_i_in = 0
                gain = 0
                cata = node_neigh.catagory
                for id_c_1 in com_graph.partion[cata]:
                    for id_c_2 in com_graph.partion[cata]:
                        if id_c_1 == id_c_2:
                            continue
                        if id_c_2 in com_graph.node_dict[id_c_1].edge_id:
                            sum_in+=0.5
                for id_c in com_graph.partion[cata]:
                    sum_total += com_graph.node_dict[id_c].k
                sum_total -=sum_in
                for id_node in node.edge_id :
                    if id_node in com_graph.partion[cata]:
                        k_i_in +=1
                
                gain = (sum_in+k_i_in)/(2*m)-((sum_total+k_i)/(2*m))*((sum_total+k_i)/(2*m))- (sum_in/(2*m)-(sum_total/(2*m))*(sum_total/(2*m))-(k_i/(2*m))*(k_i/(2*m)))                        
                if gain > max_gain:
                    max_gain = gain
                    max_cata = cata
            if max_cata == node.catagory:
                continue
            else:
                index_cata_old = com_graph.partion[node.catagory].index(ID)
                com_graph.partion[node.catagory].pop(index_cata_old)
                com_graph.partion[max_cata].append(ID)
                node.catagory = max_cata
                flag2 = 1
     '''          
                
                    
                    
                    
                