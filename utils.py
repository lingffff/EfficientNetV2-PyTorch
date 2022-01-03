# coding: utf-8
# Author: lingff (ling@stu.pku.edu.cn)
# Description: For EfficientNet V2 utils.
# Create: 2021-12-2

import re
import collections


################################################################################
# Helper functions for loading model params
################################################################################

# BlockDecoder: A Class for encoding and decoding BlockArgs
# efficientnet_params: A function to query compound coefficient
# get_model_params and efficientnet:
#     Functions to get BlockArgs and GlobalParams for efficientnet
# url_map and url_map_advprop: Dicts of url_map for pretrained weights
# load_pretrained_weights: A function to load pretrained weights

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
    'input_filters', 'output_filters', 'se_ratio', 'id_skip', 'fused'])

class BlockDecoder(object):
    """Block Decoder for readability,
       straight from the official TensorFlow repository.
    """

    @staticmethod
    def _decode_block_string(block_string):
        """Get a block through a string notation of arguments.

        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.

        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            num_repeat=int(options['r']),
            kernel_size=int(options['k']),
            stride=[int(options['s'][0])],
            expand_ratio=int(options['e']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            se_ratio=float(options['se']) if 'se' in options else None,
            fused=('f' in block_string),
            id_skip=('noskip' not in block_string))

    @staticmethod
    def decode(string_list):
        """Decode a list of string notations to specify blocks inside the network.

        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.

        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def get_efficientnetv2_params(model_name, num_classes):
    #################### EfficientNet V2 configs ####################
    v2_base_block = [  # The baseline config for v2 models.
        'r1_k3_s1_e1_i32_o16_f',
        'r2_k3_s2_e4_i16_o32_f',
        'r2_k3_s2_e4_i32_o48_f',
        'r3_k3_s2_e4_i48_o96_se0.25',
        'r5_k3_s1_e6_i96_o112_se0.25',
        'r8_k3_s2_e6_i112_o192_se0.25',
    ]
    v2_s_block = [  # about base * (width1.4, depth1.8)
        'r2_k3_s1_e1_i24_o24_f',
        'r4_k3_s2_e4_i24_o48_f',
        'r4_k3_s2_e4_i48_o64_f',
        'r6_k3_s2_e4_i64_o128_se0.25',
        'r9_k3_s1_e6_i128_o160_se0.25',
        'r15_k3_s2_e6_i160_o256_se0.25',
    ]
    v2_m_block = [  # about base * (width1.6, depth2.2)
        'r3_k3_s1_e1_i24_o24_f',
        'r5_k3_s2_e4_i24_o48_f',
        'r5_k3_s2_e4_i48_o80_f',
        'r7_k3_s2_e4_i80_o160_se0.25',
        'r14_k3_s1_e6_i160_o176_se0.25',
        'r18_k3_s2_e6_i176_o304_se0.25',
        'r5_k3_s1_e6_i304_o512_se0.25',
    ]
    v2_l_block = [  # about base * (width2.0, depth3.1)
        'r4_k3_s1_e1_i32_o32_f',
        'r7_k3_s2_e4_i32_o64_f',
        'r7_k3_s2_e4_i64_o96_f',
        'r10_k3_s2_e4_i96_o192_se0.25',
        'r19_k3_s1_e6_i192_o224_se0.25',
        'r25_k3_s2_e6_i224_o384_se0.25',
        'r7_k3_s1_e6_i384_o640_se0.25',
    ]
    v2_xl_block = [  # only for 21k pretraining.
        'r4_k3_s1_e1_i32_o32_f',
        'r8_k3_s2_e4_i32_o64_f',
        'r8_k3_s2_e4_i64_o96_f',
        'r16_k3_s2_e4_i96_o192_se0.25',
        'r24_k3_s1_e6_i192_o256_se0.25',
        'r32_k3_s2_e6_i256_o512_se0.25',
        'r8_k3_s1_e6_i512_o640_se0.25',
    ]

    efficientnetv2_params = {
        # (block, width, depth, train_size, eval_size, dropout, randaug, mixup, aug)
        'efficientnetv2-s':  # 83.9% @ 22M
            (v2_s_block, 1.0, 1.0, 300, 384, 0.2, 10, 0, 'randaug'),
        'efficientnetv2-m':  # 85.2% @ 54M
            (v2_m_block, 1.0, 1.0, 384, 480, 0.3, 15, 0.2, 'randaug'),
        'efficientnetv2-l':  # 85.7% @ 120M
            (v2_l_block, 1.0, 1.0, 384, 480, 0.4, 20, 0.5, 'randaug'),

        'efficientnetv2-xl':
            (v2_xl_block, 1.0, 1.0, 384, 512, 0.4, 20, 0.5, 'randaug'),

        # For fair comparison to EfficientNetV1, using the same scaling and autoaug.
        'efficientnetv2-b0':  # 78.7% @ 7M params
            (v2_base_block, 1.0, 1.0, 192, 224, 0.2, 0, 0, 'effnetv1_autoaug'),
        'efficientnetv2-b1':  # 79.8% @ 8M params
            (v2_base_block, 1.0, 1.1, 192, 240, 0.2, 0, 0, 'effnetv1_autoaug'),
        'efficientnetv2-b2':  # 80.5% @ 10M params
            (v2_base_block, 1.1, 1.2, 208, 260, 0.3, 0, 0, 'effnetv1_autoaug'),
        'efficientnetv2-b3':  # 82.1% @ 14M params
            (v2_base_block, 1.2, 1.4, 240, 300, 0.3, 0, 0, 'effnetv1_autoaug'),
    }

    assert model_name in list(efficientnetv2_params.keys()), "Wrong model name."
    all_params = efficientnetv2_params[model_name]

    blocks_args = BlockDecoder.decode(all_params[0])

    global_params = GlobalParams(
        width_coefficient=all_params[1],
        depth_coefficient=all_params[2],
        image_size=all_params[3],
        dropout_rate=all_params[5],
        num_classes=num_classes,

        batch_norm_momentum=None, #0.99,
        batch_norm_epsilon=None, #1e-3,
        drop_connect_rate=None, #drop_connect_rate,
        depth_divisor=None, #8,
        min_depth=None, #None,
        include_top=None, #include_top,
    )

    return blocks_args, global_params

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# # Myself, not good
# def accuracy(preds, targets):
#     with torch.no_grad():
#         batch_size = targets.size(0)
#         pred_index = torch.argmax(preds, dim=1)
#         correct = pred_index.eq(targets.view(1, -1))
#         acc1 = correct.sum().float().mul_(100.0 / batch_size)
#     return acc1

# good
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


if __name__ == '__main__':
    b, g = get_efficientnetv2_params("efficientnetv2-s", 10)
    print(b[0].input_filters)