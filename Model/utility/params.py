import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Model Params.")
    #数据读取
    parser.add_argument('--data_path', nargs='?', default='../Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-10m',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    #是否使用预训练模型
    parser.add_argument('--pretrain', type=int, default=0,
                        help='0: No pretrain, 1:Use stored models.')
    parser.add_argument('--embed_name', nargs='?', default='',
                        help='Name for pretrained model.')
    #Training Stage
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--regs', nargs='?', default='[1e-4,1e-4,1e-4]',
                        help='Regularizations.')
    #模型尺寸
    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('-mess_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--embed_size', type=int, default=50,
                        help='Embedding size.')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Layer numbers.')

    parser.add_argument('--ssl_ratio', type=float, default=0.1,
                        help='ssl_ratio.')
    parser.add_argument('--ssl_temp', type=float, default=1.5,
                        help='ssl_temp.')
    parser.add_argument('--ssl_reg', type=float, default=0.05,
                        help='ssl_temp.')

    #测试阶段
    parser.add_argument('--show_step', type=int, default=3,
                        help='Test every show_step epochs.')
    parser.add_argument('--early', type=int, default=5,
                        help='Step for stopping')
    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Metrics scale')
    #保存模型
    parser.add_argument('--save_flag', type=int, default=0,
                        help='0: Disable model saver, 1: Save Better Model')
    parser.add_argument('--save_name', nargs='?', default='best_model',
                        help='Save_name.')
    #测试规模
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    return parser.parse_args()
