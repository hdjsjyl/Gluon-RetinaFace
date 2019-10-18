



def generate_config(is_train):
    class General:
        log_frequency = 10
        name          = __name__.rsplit('/')[-1].rsplit('.')[-1]
        network       = 'resnet'
        depth         = 50
        batch_image   = 8 if is_train else 1
        strides       = [4, 8, 16, 32, 64]
        base_size     = 16
        ratios        = (1.0, )
        ## scales = 2.0**(1.0/3)
        scales        = [1.0, 1.26, 1.59, 2.0, 2.52, 3.17, 4.0, 5.04, 6.35, 8.0, 10.08, 12.70, 16.0, 20.16, 25.40]
        image_roi     = 256

    class KvstoreParam:
        kvstore     = 'device'
        batch_image = General.batch_image
        gpus        = [0]

    class NormalizeParam:
        normalizer = None

    class BackboneParam:
        normalizer = NormalizeParam.normalizer

    class NeckParam:
        normalizer = NormalizeParam.normalizer

    class RpnParam:
        num_class   = 1 + 1
        normalizer  = NormalizeParam.normalizer
        batch_image = General.batch_image

        class anchor_generate:
            scales = General.scales
            ratios = General.ratios
            strides = General.strides
            image_anchor = None

        class head:
            conv_channel = 256
            mean         = None
            std          = None

        class proposal:
            pre_nms_top_n  = 1000
            post_nms_top_n = None
            nms_thr        = None
            min_bbox_side  = None
            min_det_score  = 0.05 ## filter score in network

        class subsample_proposal:
            proposal_wo_gt = None
            image_roi      = None
            fg_fraction    = 0.25
            fg_thr         = 0.5
            bg_thr_hi      = 0.3
            bg_thr_lo      = None

        class bbox_target:
            num_reg_class  = None
            class_agnostic = None
            weight         = None
            mean           = None
            std            = None

        class focal_loss:
            alpha = 0.25
            gamma = 2.0

    class BboxParam:
        normalizer  = NormalizeParam.normalizer
        num_class   = None
        image_roi   = None
        batch_image = None

        class regress_target:
            class_agnostic = None
            mean           = None
            std            = None

    class RoiParam:
        normalizer = NormalizeParam.normalizer
        out_size   = None
        stride     = None

    class DatasetParam:
        Dataset = 'widerface'
        if is_train:
            image_set = ('train', )
        else:
            image_set = ('val', )

    # backbone  = Backbone(BackboneParam)
    # neck      = Neck(NeckParam)
    # rpn_head  = RpnHead(RpnParam)
    # bbox_head = BboxHead(bbox_head)
    # detector  = Detector()
    # if is_train:
    #     train_net = detector.get_train_net(backbone, neck, rpn_head, bbox_head)
    #     test_net  = None
    # else:
    #     train_net = None
    #     test_net  = detector.get_test_net(backbone, neck, rpn_head, bbox_head)

    class ModelParam:
        train_netw = None
        test_netw  = None

        from_scratch = False
        randome      = True

        class pretrain:
            prefix      = 'model/'+General.network+'-'+str(General.depth)
            epoch       = 0
            fixed_param = ['conv0', 'stage1', 'gamma', 'beta']

    class OptimizeParam:
        class optimizer:
            type          = None
            lr            = 0.001
            wd            = None
            momentum      = None
            clip_gradient = None

        class schedule:
            begin_epoch   = 0
            end_epoch     = 40
            lr_steps     = [1, 2, 3, 4, 5, 28, 34, 40]

        class warmup:
            type = None
            lr   = 0.0
            iter = None

    class TestParam:
        min_det_score     = None
        max_det_per_image = None

        process_roidb  = lambda x: x
        process_output = lambda x, y:x

        class model:
            prefix = None
            epoch = OptimizeParam.schedule.end_epoch

        class nms:
            type = 'nms'
            thr  = 0.3

        class proposals:
            rpn_pre_nms_top_n  = 1000
            rpn_post_nms_top_n = 3000


    class ResizeParam:
        constant  = False
        rscale    = 1.0
        pre_short = 1200
        pre_long  = 1600
        short     = 640
        long      = 640

    class ColorParam:
        color_mode      = 2
        color_jittering = 0.125

    class PadParam:
        pre_short = ResizeParam.pre_short
        pre_long  = ResizeParam.pre_long
        short     = ResizeParam.short
        long      = ResizeParam.long
        max_num_gt = 2000

    class NormParam:
        mean = [0.0, 0.0, 0.0]
        std  = [1.0, 1.0, 1.0]

    class AnchorTarget2DParam:
        def __init__(self):
            self.generate = self._generate()

        class _generate():
            def __init__(self):
                self.short   = None
                self.long    = None
                self.strides = General.strides
            scales = General.scales
            ratios = General.ratios

        class assign:
            allowed_border = 9999
            pos_thr        = 0.5
            neg_thr        = 0.4
            min_pos_thr    = 0.0

        class sample:
            image_anchor = None
            pos_fraction = None

    class RenameParam:
        mapping = dict(image='data')


    # from core.detection_input import ReadRoiRecord, Resize2DImageBbox, \
    #     ConvertImageFromHwcToChw, Flip2DImageBbox, Pad2DImageBbox, RenameRecord
    # from models.retinaface.input import PyramidAnchorTarget2D, Norm2DImage

    # if is_train:
    #     transform = [
    #         ReadRoiRecord(None),
    #         Norm2DImage(NormParam),
    #         Resize2DImageBbox(ResizeParam),
    #         Flip2DImageBbox(),
    #         Pad2DImageBbox(PadParam),
    #         ConvertImageFromHwcToChw(),
    #         PyramidAnchorTarget2D(AnchorTarget2DParam),
    #         RenameRecord(RenameParam.mapping)
    #     ]
    #     data_name  = None
    #     label_name = None
    #
    # else:
    #     transform = [
    #         ReadRoiRecord(None),
    #         Norm2DImage(NormParam),
    #         Resize2DImageBbox(ResizeParam),
    #         ConvertImageFromHwcToChw(),
    #         RenameRecord(RenameParam.mapping)
    #     ]
    #     data_name  = None
    #     label_name = None
    #
    # from models.retinaface import metric
    #
    # rpn_acc_metric = metric.FGAccMetric(
    #     'FGAcc',
    #     ['cls_loss_output'],
    #     ['rpn_cls_label']
    # )
    #
    # metric_list = [rpn_acc_metric]
    transform = None
    data_name = None
    label_name = None
    metric_list = None

    return General, KvstoreParam, RpnParam, RoiParam, BboxParam, DatasetParam, ModelParam, \
           OptimizeParam, TestParam, transform, data_name, label_name, metric_list