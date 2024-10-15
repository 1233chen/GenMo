from options.base_vae_options import BaseOptions
# import argparse

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # self.parser.add_argument('--dim_dis_hid', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--lambda_rec', type=float, default=1.0, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--lambda_rec_c', type=float, default=1.0, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--lambda_kld_gl', type=float, default=0.1, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--lambda_kld_gl12', type=float, default=0.5, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--lambda_kld', type=float, default=0.001, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--lambda_sms', type=float, default=0.001, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--lambda_spa', type=float, default=0.001, help='Dimension of hidden unit in GRU')


        self.parser.add_argument('--lambda_adv', type=float, default=1, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--lambda_R1', type=float, default=100, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--adv_start_ep', type=int, default=0, help='Dimension of hidden unit in GRU')


        self.parser.add_argument('--adv_saturate', action="store_true", help='models are saved here')
        # self.parser.add_argument('--adv_mode', type=str, required=True, help='models are saved here')


        self.parser.add_argument('--lr', type=float, default=1e-4, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--log_every', type=int, default=5, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--save_latest', type=int, default=50, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--save_every_e', type=int, default=200, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--eval_every_e', type=int, default=100, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='models are saved here')
        self.parser.add_argument('--feat_bias', type=float, default=1, help='Layers of GRU')

        self.parser.add_argument('--batch_size', type=int, default=16, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--max_epoch', type=int, default=30000, help='Dimension of hidden unit in GRU')

        self.is_train = True

class TrainRegressorOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # self.parser.add_argument('--dim_dis_hid', type=int, default=512, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--noise_scale', type=float, default=0.1,
                                 help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--noise_prob', type=float, default=0.5,
                                 help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--lr', type=float, default=2e-4, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--is_style', action="store_true", help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--is_content', action="store_true", help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--negative_margin', type=float, default=10, help='Dimension of hidden unit in GRU')


        self.parser.add_argument('--log_every', type=int, default=5, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--save_latest', type=int, default=50, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--save_every_e', type=int, default=200, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--eval_every_e', type=int, default=100, help='Dimension of hidden unit in GRU')

        self.parser.add_argument('--is_continue', action="store_true", help='models are saved here')
        self.parser.add_argument('--batch_size', type=int, default=16, help='Dimension of hidden unit in GRU')
        self.parser.add_argument('--max_epoch', type=int, default=30000, help='Dimension of hidden unit in GRU')

        self.is_train = True
