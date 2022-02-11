Log_Name='ResNet50_CropNet_transferToTargetDomain_FER2013toExpW'
Resume_Model='./exp/1642263039/ResNet50_CropNet_trainOnSourceDomain_FER2013toExpW.pkl'
OutputPath='./exp'
GPU_ID=0
Backbone='ResNet50'
useAFN='False'
methodOfAFN='SAFN'
radius=25
deltaRadius=1
weight_L2norm=0.05
useDAN='True'
methodOfDAN='CDAN-E'
faceScale=112
sourceDataset='FER2013'
targetDataset='ExpW'
train_batch_size=32
test_batch_size=32
useMultiDatasets='False'
epochs=60
lr=0.00001
lr_ad=0.001
momentum=0.9
weight_decay=0.0005
isTest='False'
showFeature='False'
class_num=7
num_divided=10
useIntraGCN='False'
useInterGCN='False'
useLocalFeature='True'
useRandomMatrix='False'
useAllOneMatrix='False'
useCov='False'
useCluster='False'
target_loss_ratio=5

OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=${GPU_ID} python TransferToTargetDomain.py\
    --Log_Name ${Log_Name}\
    --OutputPath ${OutputPath}\
    --Backbone ${Backbone}\
    --Resume_Model ${Resume_Model}\
    --GPU_ID ${GPU_ID}\
    --useAFN ${useAFN}\
    --methodOfAFN ${methodOfAFN}\
    --radius ${radius}\
    --deltaRadius ${deltaRadius}\
    --weight_L2norm ${weight_L2norm}\
    --useDAN ${useDAN}\
    --methodOfDAN ${methodOfDAN}\
    --faceScale ${faceScale}\
    --sourceDataset ${sourceDataset}\
    --targetDataset ${targetDataset}\
    --train_batch_size ${train_batch_size}\
    --test_batch_size ${test_batch_size}\
    --useMultiDatasets ${useMultiDatasets}\
    --epochs ${epochs}\
    --lr ${lr}\
    --lr_ad ${lr_ad}\
    --momentum ${momentum}\
    --weight_decay ${weight_decay}\
    --isTest ${isTest}\
    --showFeature ${showFeature}\
    --class_num ${class_num}\
    --num_divided ${num_divided}\
    --useIntraGCN ${useIntraGCN}\
    --useInterGCN ${useInterGCN}\
    --useLocalFeature ${useLocalFeature}\
    --useRandomMatrix ${useRandomMatrix}\
    --useAllOneMatrix ${useAllOneMatrix}\
    --useCov ${useCov}\
    --useCluster ${useCluster}\
    --target_loss_ratio ${target_loss_ratio}