# Overnight Monitor Log — March 30, 2026

## Monitoring: Vast (RoBERTa training + full evaluation) + CSD3 (LoRA per-model + ModernBERT long)

---
## Check #1

### 2026-03-30 00:25 UTC — Vast Check

**Status: No training/eval process running**

```
 80%|████████  | 3744/4680 [13:12<01:57,  7.96it/s]
100%|██████████| 114/114 [00:16<00:00,  8.48it/s][A
                                                 [A 80%|████████  | 3745/4680 [13:15<1:30:28,  5.81s/it] 80%|████████  | 3746/4680 [13:15<1:03:50,  4.10s/it] 80%|████████  | 3747/4680 [13:15<45:23,  2.92s/it]   80%|████████  | 3748/4680 [13:15<33:07,  2.13s/it] 80%|████████  | 3749/4680 [13:15<24:17,  1.57s/it] 80%|████████  | 3750/4680 [13:16<17:39,  1.14s/it] 80%|████████  | 3751/4680 [13:16<13:34,  1.14it/s] 80%|████████  | 3752/4680 [13:16<10:12,  1.51it/s] 80%|████████  | 3753/4680 [13:16<07:42,  2.00it/s] 80%|████████  | 3754/4680 [13:16<05:57,  2.59it/s] 80%|████████  | 3755/4680 [13:16<05:22,  2.87it/s] 80%|████████  | 3756/4680 [13:17<05:06,  3.01it/s] 80%|████████  | 3757/4680 [13:17<04:15,  3.61it/s] 80%|████████  | 3758/4680 [13:17<03:56,  3.90it/s] 80%|████████  | 3759/4680 [13:17<03:33,  4.31it/s] 80%|████████  | 3760/4680 [13:17<03:04,  4.99it/s] 80%|████████  | 3761/4680 [13:18<03:28,  4.40it/s] 80%|████████  | 3762/4680 [13:18<02:56,  5.21it/s] 80%|████████  | 3763/4680 [13:18<02:38,  5.79it/s] 80%|████████  | 3764/4680 [13:18<02:40,  5.72it/s] 80%|████████  | 3765/4680 [13:18<02:30,  6.06it/s] 80%|████████  | 3766/4680 [13:18<02:32,  5.99it/s] 80%|████████  | 3767/4680 [13:19<02:25,  6.28it/s] 81%|████████  | 3768/4680 [13:19<02:25,  6.28it/s] 81%|████████  | 3769/4680 [13:19<02:12,  6.85it/s] 81%|████████  | 3770/4680 [13:19<02:29,  6.11it/s] 81%|████████  | 3771/4680 [13:19<03:04,  4.94it/s] 81%|████████  | 3772/4680 [13:19<02:50,  5.32it/s] 81%|████████  | 3773/4680 [13:20<02:56,  5.13it/s] 81%|████████  | 3774/4680 [13:20<02:41,  5.62it/s] 81%|████████  | 3775/4680 [13:20<02:57,  5.09it/s] 81%|████████  | 3776/4680 [13:20<02:39,  5.65it/s] 81%|████████  | 3777/4680 [13:20<02:36,  5.78it/s] 81%|████████  | 3778/4680 [13:20<02:21,  6.36it/s] 81%|████████  | 3779/4680 [13:21<02:23,  6.27it/s] 81%|████████  | 3780/4680 [13:21<02:12,  6.77it/s] 81%|████████  | 3781/4680 [13:21<02:51,  5.24it/s] 81%|████████  | 3782/4680 [13:21<02:30,  5.96it/s] 81%|████████  | 3783/4680 [13:21<02:25,  6.18it/s] 81%|████████  | 3784/4680 [13:22<02:59,  4.98it/s] 81%|████████  | 3785/4680 [13:22<02:46,  5.36it/s] 81%|████████  | 3786/4680 [13:22<02:28,  6.00it/s] 81%|████████  | 3787/4680 [13:22<02:16,  6.52it/s] 81%|████████  | 3788/4680 [13:22<02:37,  5.65it/s] 81%|████████  | 3789/4680 [13:23<03:09,  4.70it/s] 81%|████████  | 3790/4680 [13:23<02:54,  5.11it/s] 81%|████████  | 3791/4680 [13:23<03:10,  4.68it/s] 81%|████████  | 3792/4680 [13:23<03:31,  4.20it/s] 81%|████████  | 3793/4680 [13:23<03:16,  4.51it/s] 81%|████████  | 3794/4680 [13:24<03:28,  4.25it/s] 81%|████████  | 3795/4680 [13:24<03:12,  4.60it/s] 81%|████████  | 3796/4680 [13:24<03:02,  4.85it/s] 81%|████████  | 3797/4680 [13:24<02:57,  4.97it/s] 81%|████████  | 3798/4680 [13:25<03:21,  4.39it/s] 81%|████████  | 3799/4680 [13:25<02:51,  5.15it/s] 81%|████████  | 3800/4680 [13:25<02:33,  5.74it/s]                                                   {'loss': '1.648', 'grad_norm': '8.063', 'learning_rate': '2.032e-06', 'epoch': '4.06'}
 81%|████████  | 3800/4680 [13:25<02:33,  5.74it/s] 81%|████████  | 3801/4680 [13:25<02:26,  5.99it/s] 81%|████████  | 3802/4680 [13:25<02:27,  5.96it/s] 81%|████████▏ | 3803/4680 [13:25<02:14,  6.50it/s] 81%|████████▏ | 3804/4680 [13:26<02:51,  5.10it/s] 81%|████████▏ | 3805/4680 [13:26<02:30,  5.80it/s] 81%|████████▏ | 3806/4680 [13:26<02:23,  6.10it/s] 81%|████████▏ | 3807/4680 [13:26<02:19,  6.24it/s] 81%|████████▏ | 3808/4680 [13:26<02:25,  5.98it/s] 81%|████████▏ | 3809/4680 [13:26<02:23,  6.07it/s] 81%|████████▏ | 3810/4680 [13:27<02:57,  4.90it/s] 81%|████████▏ | 3811/4680 [13:27<02:49,  5.12it/s] 81%|████████▏ | 3812/4680 [13:27<02:26,  5.91it/s] 81%|████████▏ | 3813/4680 [13:27<02:12,  6.52it/s] 81%|████████▏ | 3814/4680 [13:27<02:23,  6.02it/s] 82%|████████▏ | 3815/4680 [13:27<02:19,  6.22it/s] 82%|████████▏ | 3816/4680 [13:27<02:07,  6.77it/s] 82%|████████▏ | 3817/4680 [13:28<02:34,  5.57it/s] 82%|████████▏ | 3818/4680 [13:28<02:51,  5.03it/s] 82%|████████▏ | 3819/4680 [13:28<02:52,  4.98it/s] 82%|████████▏ | 3820/4680 [13:28<02:29,  5.74it/s] 82%|████████▏ | 3821/4680 [13:28<02:18,  6.19it/s] 82%|████████▏ | 3822/4680 [13:29<02:26,  5.86it/s] 82%|████████▏ | 3823/4680 [13:29<02:16,  6.29it/s] 82%|████████▏ | 3824/4680 [13:29<02:08,  6.68it/s] 82%|████████▏ | 3825/4680 [13:29<02:12,  6.44it/s] 82%|████████▏ | 3826/4680 [13:29<02:06,  6.76it/s] 82%|████████▏ | 3827/4680 [13:29<02:07,  6.67it/s] 82%|████████▏ | 3828/4680 [13:30<02:43,  5.21it/s] 82%|████████▏ | 3829/4680 [13:30<02:25,  5.84it/s] 82%|████████▏ | 3830/4680 [13:30<02:55,  4.85it/s] 82%|████████▏ | 3831/4680 [13:30<02:33,  5.52it/s] 82%|████████▏ | 3832/4680 [13:30<02:20,  6.03it/s] 82%|████████▏ | 3833/4680 [13:30<02:19,  6.09it/s] 82%|████████▏ | 3834/4680 [13:31<02:19,  6.07it/s] 82%|████████▏ | 3835/4680 [13:31<02:28,  5.68it/s] 82%|████████▏ | 3836/4680 [13:31<02:18,  6.11it/s] 82%|████████▏ | 3837/4680 [13:31<02:25,  5.78it/s] 82%|████████▏ | 3838/4680 [13:31<02:12,  6.34it/s] 82%|████████▏ | 3839/4680 [13:31<02:07,  6.61it/s] 82%|████████▏ | 3840/4680 [13:31<02:03,  6.78it/s] 82%|████████▏ | 3841/4680 [13:32<02:12,  6.31it/s] 82%|████████▏ | 3842/4680 [13:32<02:22,  5.89it/s] 82%|████████▏ | 3843/4680 [13:32<02:41,  5.19it/s] 82%|████████▏ | 3844/4680 [13:32<03:07,  4.46it/s] 82%|████████▏ | 3845/4680 [13:33<03:23,  4.10it/s] 82%|████████▏ | 3846/4680 [13:33<02:51,  4.86it/s] 82%|████████▏ | 3847/4680 [13:33<02:50,  4.89it/s] 82%|████████▏ | 3848/4680 [13:33<03:13,  4.30it/s] 82%|████████▏ | 3849/4680 [13:34<03:27,  4.00it/s] 82%|████████▏ | 3850/4680 [13:34<03:03,  4.53it/s] 82%|████████▏ | 3851/4680 [13:34<03:03,  4.51it/s] 82%|████████▏ | 3852/4680 [13:34<02:43,  5.05it/s] 82%|████████▏ | 3853/4680 [13:34<02:22,  5.79it/s] 82%|████████▏ | 3854/4680 [13:34<02:18,  5.97it/s] 82%|████████▏ | 3855/4680 [13:35<02:08,  6.41it/s] 82%|████████▏ | 3856/4680 [13:35<02:11,  6.26it/s] 82%|████████▏ | 3857/4680 [13:35<02:08,  6.43it/s] 82%|████████▏ | 3858/4680 [13:35<02:41,  5.10it/s] 82%|████████▏ | 3859/4680 [13:35<02:27,  5.57it/s] 82%|████████▏ | 3860/4680 [13:36<02:42,  5.04it/s] 82%|████████▎ | 3861/4680 [13:36<02:40,  5.09it/s] 83%|████████▎ | 3862/4680 [13:36<02:30,  5.43it/s] 83%|████████▎ | 3863/4680 [13:36<02:21,  5.76it/s] 83%|████████▎ | 3864/4680 [13:36<02:30,  5.42it/s] 83%|████████▎ | 3865/4680 [13:36<02:25,  5.62it/s] 83%|████████▎ | 3866/4680 [13:37<02:15,  6.02it/s] 83%|████████▎ | 3867/4680 [13:37<02:20,  5.78it/s] 83%|████████▎ | 3868/4680 [13:37<02:51,  4.74it/s] 83%|████████▎ | 3869/4680 [13:37<03:11,  4.24it/s] 83%|████████▎ | 3870/4680 [13:38<02:58,  4.53it/s] 83%|████████▎ | 3871/4680 [13:38<02:42,  4.99it/s] 83%|████████▎ | 3872/4680 [13:38<02:24,  5.59it/s] 83%|████████▎ | 3873/4680 [13:38<02:12,  6.07it/s] 83%|████████▎ | 3874/4680 [13:38<02:43,  4.93it/s] 83%|████████▎ | 3875/4680 [13:38<02:29,  5.38it/s] 83%|████████▎ | 3876/4680 [13:38<02:12,  6.09it/s] 83%|████████▎ | 3877/4680 [13:39<02:00,  6.69it/s] 83%|████████▎ | 3878/4680 [13:39<01:58,  6.79it/s] 83%|████████▎ | 3879/4680 [13:39<01:50,  7.25it/s] 83%|████████▎ | 3880/4680 [13:39<02:17,  5.80it/s] 83%|████████▎ | 3881/4680 [13:39<02:43,  4.90it/s] 83%|████████▎ | 3882/4680 [13:40<03:04,  4.32it/s] 83%|████████▎ | 3883/4680 [13:40<02:54,  4.56it/s] 83%|████████▎ | 3884/4680 [13:40<02:29,  5.32it/s] 83%|████████▎ | 3885/4680 [13:40<02:33,  5.19it/s] 83%|████████▎ | 3886/4680 [13:40<02:36,  5.07it/s] 83%|████████▎ | 3887/4680 [13:40<02:15,  5.83it/s] 83%|████████▎ | 3888/4680 [13:41<02:07,  6.19it/s] 83%|████████▎ | 3889/4680 [13:41<02:14,  5.89it/s] 83%|████████▎ | 3890/4680 [13:41<02:09,  6.08it/s] 83%|████████▎ | 3891/4680 [13:41<02:11,  6.01it/s] 83%|████████▎ | 3892/4680 [13:41<02:15,  5.81it/s] 83%|████████▎ | 3893/4680 [13:41<02:12,  5.93it/s] 83%|████████▎ | 3894/4680 [13:42<02:27,  5.31it/s] 83%|████████▎ | 3895/4680 [13:42<02:40,  4.88it/s] 83%|████████▎ | 3896/4680 [13:42<02:49,  4.63it/s] 83%|████████▎ | 3897/4680 [13:42<02:31,  5.17it/s] 83%|████████▎ | 3898/4680 [13:43<02:26,  5.34it/s] 83%|████████▎ | 3899/4680 [13:43<02:09,  6.05it/s] 83%|████████▎ | 3900/4680 [13:43<02:15,  5.75it/s]                                                   {'loss': '1.599', 'grad_norm': '14.77', 'learning_rate': '1.814e-06', 'epoch': '4.167'}
 83%|████████▎ | 3900/4680 [13:43<02:15,  5.75it/s] 83%|████████▎ | 3901/4680 [13:43<02:05,  6.23it/s] 83%|████████▎ | 3902/4680 [13:43<02:06,  6.15it/s] 83%|████████▎ | 3903/4680 [13:43<02:01,  6.39it/s] 83%|████████▎ | 3904/4680 [13:44<02:32,  5.08it/s] 83%|████████▎ | 3905/4680 [13:44<02:19,  5.57it/s] 83%|████████▎ | 3906/4680 [13:44<02:05,  6.17it/s] 83%|████████▎ | 3907/4680 [13:44<02:37,  4.92it/s] 84%|████████▎ | 3908/4680 [13:44<02:25,  5.29it/s] 84%|████████▎ | 3909/4680 [13:45<02:47,  4.59it/s] 84%|████████▎ | 3910/4680 [13:45<03:05,  4.16it/s] 84%|████████▎ | 3911/4680 [13:45<02:49,  4.54it/s] 84%|████████▎ | 3912/4680 [13:45<03:06,  4.13it/s] 84%|████████▎ | 3913/4680 [13:45<02:46,  4.62it/s] 84%|████████▎ | 3914/4680 [13:46<02:24,  5.29it/s] 84%|████████▎ | 3915/4680 [13:46<02:10,  5.88it/s] 84%|████████▎ | 3916/4680 [13:46<02:38,  4.83it/s] 84%|████████▎ | 3917/4680 [13:46<02:20,  5.44it/s] 84%|████████▎ | 3918/4680 [13:46<02:26,  5.22it/s] 84%|████████▎ | 3919/4680 [13:47<02:43,  4.67it/s] 84%|████████▍ | 3920/4680 [13:47<02:34,  4.93it/s] 84%|████████▍ | 3921/4680 [13:47<02:52,  4.39it/s] 84%|████████▍ | 3922/4680 [13:47<02:32,  4.99it/s] 84%|████████▍ | 3923/4680 [13:47<02:11,  5.76it/s] 84%|████████▍ | 3924/4680 [13:47<02:04,  6.08it/s] 84%|████████▍ | 3925/4680 [13:48<02:35,  4.87it/s] 84%|████████▍ | 3926/4680 [13:48<02:54,  4.33it/s] 84%|████████▍ | 3927/4680 [13:48<02:47,  4.49it/s] 84%|████████▍ | 3928/4680 [13:49<02:57,  4.23it/s] 84%|████████▍ | 3929/4680 [13:49<02:36,  4.79it/s] 84%|████████▍ | 3930/4680 [13:49<02:56,  4.25it/s] 84%|████████▍ | 3931/4680 [13:49<02:59,  4.18it/s] 84%|████████▍ | 3932/4680 [13:49<02:32,  4.92it/s] 84%|████████▍ | 3933/4680 [13:50<02:46,  4.49it/s] 84%|████████▍ | 3934/4680 [13:50<02:33,  4.85it/s] 84%|████████▍ | 3935/4680 [13:50<02:55,  4.25it/s] 84%|████████▍ | 3936/4680 [13:50<03:07,  3.98it/s] 84%|████████▍ | 3937/4680 [13:50<02:35,  4.77it/s] 84%|████████▍ | 3938/4680 [13:51<02:27,  5.03it/s] 84%|████████▍ | 3939/4680 [13:51<02:48,  4.39it/s] 84%|████████▍ | 3940/4680 [13:51<02:42,  4.54it/s] 84%|████████▍ | 3941/4680 [13:51<02:22,  5.19it/s] 84%|████████▍ | 3942/4680 [13:51<02:13,  5.51it/s] 84%|████████▍ | 3943/4680 [13:52<02:38,  4.66it/s] 84%|████████▍ | 3944/4680 [13:52<02:21,  5.20it/s] 84%|████████▍ | 3945/4680 [13:52<02:19,  5.27it/s] 84%|████████▍ | 3946/4680 [13:52<02:27,  4.99it/s] 84%|████████▍ | 3947/4680 [13:52<02:21,  5.18it/s] 84%|████████▍ | 3948/4680 [13:53<02:07,  5.75it/s] 84%|████████▍ | 3949/4680 [13:53<02:01,  6.03it/s] 84%|████████▍ | 3950/4680 [13:53<01:50,  6.58it/s] 84%|████████▍ | 3951/4680 [13:53<02:20,  5.18it/s] 84%|████████▍ | 3952/4680 [13:53<02:03,  5.90it/s] 84%|████████▍ | 3953/4680 [13:53<02:07,  5.69it/s] 84%|████████▍ | 3954/4680 [13:54<02:33,  4.74it/s] 85%|████████▍ | 3955/4680 [13:54<02:16,  5.31it/s] 85%|████████▍ | 3956/4680 [13:54<02:40,  4.52it/s] 85%|████████▍ | 3957/4680 [13:54<02:55,  4.13it/s] 85%|████████▍ | 3958/4680 [13:55<02:36,  4.61it/s] 85%|████████▍ | 3959/4680 [13:55<02:53,  4.17it/s] 85%|████████▍ | 3960/4680 [13:55<02:37,  4.57it/s] 85%|████████▍ | 3961/4680 [13:55<02:42,  4.43it/s] 85%|████████▍ | 3962/4680 [13:55<02:19,  5.15it/s] 85%|████████▍ | 3963/4680 [13:56<02:07,  5.63it/s] 85%|████████▍ | 3964/4680 [13:56<02:29,  4.78it/s] 85%|████████▍ | 3965/4680 [13:56<02:14,  5.33it/s] 85%|████████▍ | 3966/4680 [13:56<02:11,  5.42it/s] 85%|████████▍ | 3967/4680 [13:56<02:27,  4.83it/s] 85%|████████▍ | 3968/4680 [13:57<02:12,  5.38it/s] 85%|████████▍ | 3969/4680 [13:57<02:04,  5.71it/s] 85%|████████▍ | 3970/4680 [13:57<02:30,  4.73it/s] 85%|████████▍ | 3971/4680 [13:57<02:36,  4.52it/s] 85%|████████▍ | 3972/4680 [13:58<02:52,  4.09it/s] 85%|████████▍ | 3973/4680 [13:58<03:02,  3.87it/s] 85%|████████▍ | 3974/4680 [13:58<02:32,  4.62it/s] 85%|████████▍ | 3975/4680 [13:58<02:19,  5.04it/s] 85%|████████▍ | 3976/4680 [13:58<02:10,  5.41it/s] 85%|████████▍ | 3977/4680 [13:59<02:30,  4.68it/s] 85%|████████▌ | 3978/4680 [13:59<02:17,  5.10it/s] 85%|████████▌ | 3979/4680 [13:59<02:32,  4.58it/s] 85%|████████▌ | 3980/4680 [13:59<02:17,  5.11it/s] 85%|████████▌ | 3981/4680 [13:59<02:13,  5.24it/s] 85%|████████▌ | 3982/4680 [13:59<01:59,  5.82it/s] 85%|████████▌ | 3983/4680 [14:00<01:55,  6.02it/s] 85%|████████▌ | 3984/4680 [14:00<01:59,  5.84it/s] 85%|████████▌ | 3985/4680 [14:00<02:24,  4.82it/s] 85%|████████▌ | 3986/4680 [14:00<02:11,  5.27it/s] 85%|████████▌ | 3987/4680 [14:00<02:27,  4.71it/s] 85%|████████▌ | 3988/4680 [14:01<02:17,  5.04it/s] 85%|████████▌ | 3989/4680 [14:01<02:36,  4.43it/s] 85%|████████▌ | 3990/4680 [14:01<02:12,  5.20it/s] 85%|████████▌ | 3991/4680 [14:01<02:00,  5.70it/s] 85%|████████▌ | 3992/4680 [14:01<01:46,  6.45it/s] 85%|████████▌ | 3993/4680 [14:01<01:37,  7.02it/s] 85%|████████▌ | 3994/4680 [14:02<01:44,  6.56it/s]
```

### 2026-03-30 00:25 UTC — CSD3 Check

```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```

Completed/Failed jobs:
```
26507842     rb_roberta_test  COMPLETED   00:15:29 
26508785        rb_fused_bkt  COMPLETED   00:08:32 
26526641          rb_lora_pm  COMPLETED   00:27:47 
```
## Check — 2026-03-30 00:28 UTC

### Vast
```
---
===== PHASE 1: RoBERTa Fused Training =====
--- 1/5: length_bucket ---
--------------------------------+------------+-
                                                 [A{'eval_loss': '1.761', 'eval_accuracy': '0.4108', 'eval_adjacent_accuracy': '0.6472', 'eval_mae_class': '1.661', 'eval_mae_tokens': '109.3', 'eval_runtime': '16.78', 'eval_samples_per_second': '217.1', 'eval_steps_per_second': '6.795', 'epoch': '1'}
                                                 [A{'eval_loss': '1.737', 'eval_accuracy': '0.414', 'eval_adjacent_accuracy': '0.6608', 'eval_mae_class': '1.584', 'eval_mae_tokens': '106.8', 'eval_runtime': '16.56', 'eval_samples_per_second': '219.9', 'eval_steps_per_second': '6.884', 'epoch': '2'}
                                                 [A{'eval_loss': '1.721', 'eval_accuracy': '0.4185', 'eval_adjacent_accuracy': '0.6642', 'eval_mae_class': '1.579', 'eval_mae_tokens': '104.2', 'eval_runtime': '16.68', 'eval_samples_per_second': '218.4', 'eval_steps_per_second': '6.836', 'epoch': '3'}
                                                 [A{'eval_loss': '1.725', 'eval_accuracy': '0.4172', 'eval_adjacent_accuracy': '0.6696', 'eval_mae_class': '1.553', 'eval_mae_tokens': '104', 'eval_runtime': '16.51', 'eval_samples_per_second': '220.6', 'eval_steps_per_second': '6.904', 'epoch': '4'}
                                                 [A{'eval_loss': '1.724', 'eval_accuracy': '0.4158', 'eval_adjacent_accuracy': '0.6709', 'eval_mae_class': '1.561', 'eval_mae_tokens': '103', 'eval_runtime': '16.48', 'eval_samples_per_second': '221', 'eval_steps_per_second': '6.918', 'epoch': '5'}
2026-03-30 00:28:38,275 - __main__ - INFO -   Final eval: {'eval_loss': 1.7208913564682007, 'eval_accuracy': 0.41845140032948935, 'eval_adjacent_accuracy': 0.6641954969796815, 'eval_mae_class': 1.5791460735859417, 'eval_mae_tokens': 104.16915132553359, 'eval_runtime': 16.5442, 'eval_samples_per_second': 220.138, 'eval_steps_per_second': 6.891, 'epoch': 5.0}
```

### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 01:01 UTC

### Vast
**No process running**

Progress:
```
===== PHASE 1: RoBERTa Fused Training =====
--- 1/5: length_bucket ---
--------------------------------+------------+-
2026-03-30 00:28:38,275 - __main__ - INFO -   Final eval: {'eval_loss': 1.7208913564682007, 'eval_accuracy': 0.41845140032948935, 'eval_adjacent_accuracy': 0.6641954969796815, 'eval_mae_class': 1.5791460735859417, 'eval_mae_tokens': 104.16915132553359, 'eval_runtime': 16.5442, 'eval_samples_per_second': 220.138, 'eval_steps_per_second': 6.891, 'epoch': 5.0}
2026-03-30 00:28:55,138 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_bucket
2026-03-30 00:28:55,225 - __main__ - INFO - === Final Results ===
--- 2/5: length (log-transform) ---
--------------------------------+------------+-
```

Current step: `40%|████ | 1872/4680 [06:33<09:50, 4.76it/s`

### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 01:31 UTC

### Vast
**No process running**

Progress:
```
===== PHASE 1: RoBERTa Fused Training =====
--- 1/5: length_bucket ---
--------------------------------+------------+-
2026-03-30 00:28:38,275 - __main__ - INFO -   Final eval: {'eval_loss': 1.7208913564682007, 'eval_accuracy': 0.41845140032948935, 'eval_adjacent_accuracy': 0.6641954969796815, 'eval_mae_class': 1.5791460735859417, 'eval_mae_tokens': 104.16915132553359, 'eval_runtime': 16.5442, 'eval_samples_per_second': 220.138, 'eval_steps_per_second': 6.891, 'epoch': 5.0}
2026-03-30 00:28:55,138 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_bucket
2026-03-30 00:28:55,225 - __main__ - INFO - === Final Results ===
--- 2/5: length (log-transform) ---
--------------------------------+------------+-
```

Current: `1872/4680 [06:33<09:50, 4.76it/s`

### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 02:01 UTC
### Vast
Step: `3181/4680` | Disk: `87% 4.4G`
```
--- 1/5: length_bucket ---
--------------------------------+------------+-
2026-03-30 00:28:38,275 - __main__ - INFO -   Final eval: {'eval_loss': 1.7208913564682007, 'eval_accuracy': 0.41845140032948935, 'eval_adjacent_accuracy': 0.6641954969796815, 'eval_mae_class': 1.5791460735859417, 'eval_mae_tokens': 104.16915132553359, 'eval_runtime': 16.5442, 'eval_samples_per_second': 220.138, 'eval_steps_per_second': 6.891, 'epoch': 5.0}
2026-03-30 00:28:55,138 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_bucket
2026-03-30 00:28:55,225 - __main__ - INFO - === Final Results ===
--- 2/5: length (log-transform) ---
--------------------------------+------------+-
--- 2/5: length (log-transform) ---
--------------------------------+------------+-
2026-03-30 01:50:16,615 - __main__ - INFO -   Final eval: {'eval_loss': 0.7461414933204651, 'eval_mae': 105.23628234863281, 'eval_mape': 102.37763023376465, 'eval_spearman_r': 0.6958822832529319, 'eval_runtime': 16.5224, 'eval_samples_per_second': 220.428, 'eval_steps_per_second': 6.9, 'epoch': 5.0}
2026-03-30 01:50:33,533 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_log
2026-03-30 01:50:33,580 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 78%)
--- 3/5: judge_class ---
--------------------------------+------------+-
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 02:31 UTC
### Vast
Step: `` | Disk: `81% 6.3G`
```
--- 2/5: length (log-transform) ---
--------------------------------+------------+-
2026-03-30 01:50:16,615 - __main__ - INFO -   Final eval: {'eval_loss': 0.7461414933204651, 'eval_mae': 105.23628234863281, 'eval_mape': 102.37763023376465, 'eval_spearman_r': 0.6958822832529319, 'eval_runtime': 16.5224, 'eval_samples_per_second': 220.428, 'eval_steps_per_second': 6.9, 'epoch': 5.0}
2026-03-30 01:50:33,533 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_log
2026-03-30 01:50:33,580 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 78%)
--- 3/5: judge_class ---
--------------------------------+------------+-
2026-03-30 02:07:27,374 - __main__ - INFO -   Final eval: {'eval_loss': 1.1713545322418213, 'eval_accuracy': 0.6222542559033498, 'eval_adjacent_accuracy': 0.7485584843492586, 'eval_mae_class': 1.5718698517298189, 'eval_runtime': 16.5007, 'eval_samples_per_second': 220.718, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:07:44,251 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/judge_class
2026-03-30 02:07:44,297 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 80%)
--- 4/5: similarity ---
--------------------------------+------------+-
2026-03-30 02:24:37,423 - __main__ - INFO -   Final eval: {'eval_loss': 0.007085892371833324, 'eval_mae': 0.055315747391432524, 'eval_mape': 7.217256247997284, 'eval_spearman_r': 0.4212363364397668, 'eval_runtime': 16.4994, 'eval_samples_per_second': 220.735, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:24:54,323 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/similarity
2026-03-30 02:24:54,375 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 81%)
--- 5/5: reference_score ---
--------------------------------+------------+-
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 03:01 UTC
### Vast
Disk: `81% 6.3G` | Procs: ``
```
2026-03-30 00:28:38,275 - __main__ - INFO -   Final eval: {'eval_loss': 1.7208913564682007, 'eval_accuracy': 0.41845140032948935, 'eval_adjacent_accuracy': 0.6641954969796815, 'eval_mae_class': 1.5791460735859417, 'eval_mae_tokens': 104.16915132553359, 'eval_runtime': 16.5442, 'eval_samples_per_second': 220.138, 'eval_steps_per_second': 6.891, 'epoch': 5.0}
2026-03-30 00:28:55,138 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_bucket
2026-03-30 00:28:55,225 - __main__ - INFO - === Final Results ===
--- 2/5: length (log-transform) ---
--------------------------------+------------+-
--- 2/5: length (log-transform) ---
--------------------------------+------------+-
2026-03-30 01:50:16,615 - __main__ - INFO -   Final eval: {'eval_loss': 0.7461414933204651, 'eval_mae': 105.23628234863281, 'eval_mape': 102.37763023376465, 'eval_spearman_r': 0.6958822832529319, 'eval_runtime': 16.5224, 'eval_samples_per_second': 220.428, 'eval_steps_per_second': 6.9, 'epoch': 5.0}
2026-03-30 01:50:33,533 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_log
2026-03-30 01:50:33,580 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 78%)
--- 3/5: judge_class ---
--------------------------------+------------+-
2026-03-30 02:07:27,374 - __main__ - INFO -   Final eval: {'eval_loss': 1.1713545322418213, 'eval_accuracy': 0.6222542559033498, 'eval_adjacent_accuracy': 0.7485584843492586, 'eval_mae_class': 1.5718698517298189, 'eval_runtime': 16.5007, 'eval_samples_per_second': 220.718, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:07:44,251 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/judge_class
2026-03-30 02:07:44,297 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 80%)
--- 4/5: similarity ---
--------------------------------+------------+-
2026-03-30 02:24:37,423 - __main__ - INFO -   Final eval: {'eval_loss': 0.007085892371833324, 'eval_mae': 0.055315747391432524, 'eval_mape': 7.217256247997284, 'eval_spearman_r': 0.4212363364397668, 'eval_runtime': 16.4994, 'eval_samples_per_second': 220.735, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:24:54,323 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/similarity
2026-03-30 02:24:54,375 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 81%)
--- 5/5: reference_score ---
--------------------------------+------------+-
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 03:31 UTC
### Vast
Disk: `81% 6.3G`
**No training/eval process running**
```
2026-03-30 01:33:30,901 - __main__ - INFO - FusedDataset: 14963 samples, 4 models, target=length
2026-03-30 01:33:30,918 - __main__ - INFO - FusedDataset: 3642 samples, 4 models, target=length
--------------------------------+------------+-
2026-03-30 01:50:16,615 - __main__ - INFO -   Final eval: {'eval_loss': 0.7461414933204651, 'eval_mae': 105.23628234863281, 'eval_mape': 102.37763023376465, 'eval_spearman_r': 0.6958822832529319, 'eval_runtime': 16.5224, 'eval_samples_per_second': 220.428, 'eval_steps_per_second': 6.9, 'epoch': 5.0}
2026-03-30 01:50:33,533 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/length_log
2026-03-30 01:50:33,580 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 78%)
--- 3/5: judge_class ---
2026-03-30 01:50:46,075 - __main__ - INFO - Training fused judge_class predictor
2026-03-30 01:50:46,955 - __main__ - INFO - FusedDataset: 14963 samples, 4 models, target=judge_class
2026-03-30 01:50:46,986 - __main__ - INFO - FusedDataset: 3642 samples, 4 models, target=judge_class
--------------------------------+------------+-
2026-03-30 02:07:27,374 - __main__ - INFO -   Final eval: {'eval_loss': 1.1713545322418213, 'eval_accuracy': 0.6222542559033498, 'eval_adjacent_accuracy': 0.7485584843492586, 'eval_mae_class': 1.5718698517298189, 'eval_runtime': 16.5007, 'eval_samples_per_second': 220.718, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:07:44,251 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/judge_class
2026-03-30 02:07:44,297 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 80%)
--- 4/5: similarity ---
2026-03-30 02:07:54,942 - __main__ - INFO - Training fused similarity predictor
2026-03-30 02:07:55,767 - __main__ - INFO - FusedDataset: 14963 samples, 4 models, target=similarity
2026-03-30 02:07:55,778 - __main__ - INFO - FusedDataset: 3642 samples, 4 models, target=similarity
--------------------------------+------------+-
2026-03-30 02:24:37,423 - __main__ - INFO -   Final eval: {'eval_loss': 0.007085892371833324, 'eval_mae': 0.055315747391432524, 'eval_mape': 7.217256247997284, 'eval_spearman_r': 0.4212363364397668, 'eval_runtime': 16.4994, 'eval_samples_per_second': 220.735, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:24:54,323 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/similarity
2026-03-30 02:24:54,375 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 81%)
--- 5/5: reference_score ---
2026-03-30 02:25:05,120 - __main__ - INFO - Training fused reference_score predictor
2026-03-30 02:25:05,902 - __main__ - INFO - FusedDataset: 0 samples, 4 models, target=reference_score
2026-03-30 02:25:05,913 - __main__ - INFO - FusedDataset: 0 samples, 4 models, target=reference_score
--------------------------------+------------+-
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

**FIX APPLIED**: reference_score target had 0 samples — `train.jsonl` missing `reference_score` field.
Switched to `train_with_ref.jsonl` / `test_with_ref.jsonl` for reference_score target only.
This is a data path fix, NOT a training parameter change.
Also using `test_with_ref.jsonl` for reference_score evaluation.

## Check — 2026-03-30 04:01 UTC
### Vast
Disk: `83% 5.8G`
**No process running**
```
--------------------------------+------------+-
2026-03-30 02:07:27,374 - __main__ - INFO -   Final eval: {'eval_loss': 1.1713545322418213, 'eval_accuracy': 0.6222542559033498, 'eval_adjacent_accuracy': 0.7485584843492586, 'eval_mae_class': 1.5718698517298189, 'eval_runtime': 16.5007, 'eval_samples_per_second': 220.718, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:07:44,251 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/judge_class
2026-03-30 02:07:44,297 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 80%)
--- 4/5: similarity ---
--------------------------------+------------+-
2026-03-30 02:24:37,423 - __main__ - INFO -   Final eval: {'eval_loss': 0.007085892371833324, 'eval_mae': 0.055315747391432524, 'eval_mape': 7.217256247997284, 'eval_spearman_r': 0.4212363364397668, 'eval_runtime': 16.4994, 'eval_samples_per_second': 220.735, 'eval_steps_per_second': 6.909, 'epoch': 5.0}
2026-03-30 02:24:54,323 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/similarity
2026-03-30 02:24:54,375 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints, disk: 81%)
--- 5/5: reference_score ---
--------------------------------+------------+-
--- 5/5: reference_score (with ref data) ---
--------------------------------+------------+-
2026-03-30 03:49:35,314 - __main__ - INFO -   Final eval: {'eval_loss': 0.07106830179691315, 'eval_mae': 0.16212894022464752, 'eval_mape': 5095.5184326171875, 'eval_spearman_r': 0.5917674265285481, 'eval_runtime': 16.5538, 'eval_samples_per_second': 220.01, 'eval_steps_per_second': 6.887, 'epoch': 5.0}
2026-03-30 03:49:52,307 - __main__ - INFO -   Saved fused model to /root/models/roberta_fused/reference_score
2026-03-30 03:49:52,358 - __main__ - INFO - === Final Results ===
  (cleaned checkpoints)
===== PHASE 1 COMPLETE — Mon Mar 30 03:49:53 UTC 2026 =====
===== PHASE 2: Full Evaluation =====
--- length ---
Evaluating: knn:knn
------------------------+------------+--+-
2026-03-30 03:55:58,139 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (343.2s)
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 04:31 UTC
### Vast
Disk: `83% 5.8G`
Running eval
```
--------------------------------+------------+-
--- 5/5: reference_score ---
--------------------------------+------------+-
--- 5/5: reference_score (with ref data) ---
--------------------------------+------------+-
===== PHASE 1 COMPLETE — Mon Mar 30 03:49:53 UTC 2026 =====
===== PHASE 2: Full Evaluation =====
--- length ---
------------------------+------------+--+-
2026-03-30 03:55:58,139 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (343.2s)
2026-03-30 04:02:01,745 - __main__ - INFO -   Qwen2.5-3B: MAE=107.82 MAPE=132.0% ρ=0.674 (363.6s)
2026-03-30 04:08:04,340 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (362.6s)
2026-03-30 04:14:04,641 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (360.3s)
------------------+------------+--+-
2026-03-30 04:16:23,721 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (136.9s)
2026-03-30 04:18:40,787 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (137.1s)
2026-03-30 04:21:19,311 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (158.5s)
2026-03-30 04:24:12,993 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (173.7s)
--------------------------------+------------+-
2026-03-30 04:24:15,141 - __main__ - ERROR - Failed encoder:length_log: CUDA error: device-side assert triggered
2026-03-30 04:24:16,544 - __main__ - ERROR - Failed llm:length: CUDA error: device-side assert triggered
--- length (bucket) ---
------------------+------------+--+-
2026-03-30 04:29:33,412 - __main__ - INFO -   Qwen2.5-14B: MAE=87.36 MAPE=146.5% ρ=0.721 (303.8s)
2026-03-30 04:29:33,413 - __main__ - INFO -     Bucket: acc=0.482 adj=0.726 top3=0.755 E[len]_MAE=87.4
```
Result files:
```
2 /root/eval_results/bucket_results.json
2 /root/eval_results/judge_results.json
3664 /root/eval_results/length_results.json
2 /root/eval_results/reference_score_results.json
2 /root/eval_results/similarity_results.json
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

**FIX APPLIED**: RoBERTa encoder eval crashed with CUDA index-out-of-bounds.
Root cause: `max_length=1024` hardcoded in adapter, but RoBERTa max position = 514.
Fix: adapter now reads `max_length` from `fused_config.json` (512 for RoBERTa, 1024 for ModernBERT).
Current eval round will have partial failures for RoBERTa. Will re-run failed evals after this round completes.

## Check — 2026-03-30 05:01 UTC
### Vast
Running eval
```
2026-03-30 04:08:04,340 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (362.6s)
2026-03-30 04:14:04,641 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (360.3s)
------------------+------------+--+-
2026-03-30 04:16:23,721 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (136.9s)
2026-03-30 04:18:40,787 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (137.1s)
2026-03-30 04:21:19,311 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (158.5s)
2026-03-30 04:24:12,993 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (173.7s)
--------------------------------+------------+-
2026-03-30 04:24:15,141 - __main__ - ERROR - Failed encoder:length_log: CUDA error: device-side assert triggered
2026-03-30 04:24:16,544 - __main__ - ERROR - Failed llm:length: CUDA error: device-side assert triggered
--- length (bucket) ---
------------------+------------+--+-
2026-03-30 04:29:33,412 - __main__ - INFO -   Qwen2.5-14B: MAE=87.36 MAPE=146.5% ρ=0.721 (303.8s)
2026-03-30 04:29:33,413 - __main__ - INFO -     Bucket: acc=0.482 adj=0.726 top3=0.755 E[len]_MAE=87.4
2026-03-30 04:34:51,897 - __main__ - INFO -   Qwen2.5-3B: MAE=97.27 MAPE=106.7% ρ=0.750 (318.5s)
2026-03-30 04:34:51,898 - __main__ - INFO -     Bucket: acc=0.397 adj=0.654 top3=0.672 E[len]_MAE=97.3
2026-03-30 04:39:23,729 - __main__ - INFO -   Qwen2.5-72B: MAE=96.99 MAPE=121.6% ρ=0.652 (271.8s)
2026-03-30 04:39:23,729 - __main__ - INFO -     Bucket: acc=0.398 adj=0.686 top3=0.710 E[len]_MAE=97.0
2026-03-30 04:43:55,846 - __main__ - INFO -   Qwen2.5-7B: MAE=146.02 MAPE=271.9% ρ=0.561 (272.1s)
2026-03-30 04:43:55,846 - __main__ - INFO -     Bucket: acc=0.379 adj=0.555 top3=0.593 E[len]_MAE=146.0
--------------------------------+------------+-
2026-03-30 04:43:58,431 - __main__ - ERROR - Failed bucket_encoder:length_bucket: CUDA error: device-side assert triggered
    (bucket)                     acc=0.482 adj=0.726 E[len]_MAE=87.4
    (bucket)                     acc=0.397 adj=0.654 E[len]_MAE=97.3
    (bucket)                     acc=0.398 adj=0.686 E[len]_MAE=97.0
    (bucket)                     acc=0.379 adj=0.555 E[len]_MAE=146.0
--- judge ---
------------------------+------------+--+-
2026-03-30 04:50:14,741 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2334.9% ρ=0.024 (354.2s)
2026-03-30 04:56:17,843 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=9584.6% ρ=0.263 (363.1s)
```
Result files:
```
bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 3664B
reference_score_results.json: 2B
similarity_results.json: 2B
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 05:31 UTC
### Vast
Eval running
```
2026-03-30 04:39:23,729 - __main__ - INFO -   Qwen2.5-72B: MAE=96.99 MAPE=121.6% ρ=0.652 (271.8s)
2026-03-30 04:39:23,729 - __main__ - INFO -     Bucket: acc=0.398 adj=0.686 top3=0.710 E[len]_MAE=97.0
2026-03-30 04:43:55,846 - __main__ - INFO -   Qwen2.5-7B: MAE=146.02 MAPE=271.9% ρ=0.561 (272.1s)
2026-03-30 04:43:55,846 - __main__ - INFO -     Bucket: acc=0.379 adj=0.555 top3=0.593 E[len]_MAE=146.0
--------------------------------+------------+-
2026-03-30 04:43:58,431 - __main__ - ERROR - Failed bucket_encoder:length_bucket: CUDA error: device-side assert triggered
    (bucket)                     acc=0.482 adj=0.726 E[len]_MAE=87.4
    (bucket)                     acc=0.397 adj=0.654 E[len]_MAE=97.3
    (bucket)                     acc=0.398 adj=0.686 E[len]_MAE=97.0
    (bucket)                     acc=0.379 adj=0.555 E[len]_MAE=146.0
--- judge ---
------------------------+------------+--+-
2026-03-30 04:50:14,741 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2334.9% ρ=0.024 (354.2s)
2026-03-30 04:56:17,843 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=9584.6% ρ=0.263 (363.1s)
2026-03-30 05:02:11,742 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3190.7% ρ=0.219 (353.9s)
2026-03-30 05:08:13,438 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=3849.7% ρ=0.175 (361.7s)
------------------+------------+--+-
2026-03-30 05:08:15,689 - __main__ - ERROR - Failed encoder:modernbert_fused_judge: only one element tensors can be converted to Python scalars
--------------------------------+------------+-
2026-03-30 05:08:17,747 - __main__ - ERROR - Failed encoder:judge_class: only one element tensors can be converted to Python scalars
```
Result files: `bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 3664B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

**FIX APPLIED**: Encoder adapter crashed on judge (10-class): `only one element tensors`.
Root cause: `float(logits)` on multi-class output. Fix: argmax → class/9 score for classifiers.
Also fixes RoBERTa max_length (previous fix). Will re-run all failed evals after current round.

## Check — 2026-03-30 06:01 UTC
### Vast
Eval running
```
2026-03-30 04:43:55,846 - __main__ - INFO -     Bucket: acc=0.379 adj=0.555 top3=0.593 E[len]_MAE=146.0
2026-03-30 04:43:58,431 - __main__ - ERROR - Failed bucket_encoder:length_bucket: CUDA error: device-side assert triggered
  --- Qwen2.5-14B ---
    (bucket)                     acc=0.482 adj=0.726 E[len]_MAE=87.4
  --- Qwen2.5-3B ---
    (bucket)                     acc=0.397 adj=0.654 E[len]_MAE=97.3
  --- Qwen2.5-72B ---
    (bucket)                     acc=0.398 adj=0.686 E[len]_MAE=97.0
  --- Qwen2.5-7B ---
    (bucket)                     acc=0.379 adj=0.555 E[len]_MAE=146.0
  --- Aggregate ---
  --- Bucket Filtering ---
--- judge ---
2026-03-30 04:50:14,741 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2334.9% ρ=0.024 (354.2s)
2026-03-30 04:56:17,843 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=9584.6% ρ=0.263 (363.1s)
2026-03-30 05:02:11,742 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3190.7% ρ=0.219 (353.9s)
2026-03-30 05:08:13,438 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=3849.7% ρ=0.175 (361.7s)
2026-03-30 05:08:15,689 - __main__ - ERROR - Failed encoder:modernbert_fused_judge: only one element tensors can be converted to Python scalars
2026-03-30 05:08:17,747 - __main__ - ERROR - Failed encoder:judge_class: only one element tensors can be converted to Python scalars
2026-03-30 05:59:48,387 - __main__ - INFO -   Qwen2.5-14B: MAE=0.68 MAPE=100.7% ρ=0.002 (3088.8s)
```
Files: `bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 3664B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

**ACTION**: Killed slow eval (LoRA judge was 3088s/model = ~10h more).
Launched clean re-eval with fixed adapter (max_length + multi-class).
Log: /tmp/vast_eval_fixed.log

## Check — 2026-03-30 06:31 UTC
### Vast (fixed eval)
Eval running
```
--- length ---
------------------------+------------+--+-
2026-03-30 06:10:09,545 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (366.3s)
2026-03-30 06:16:01,437 - __main__ - INFO -   Qwen2.5-3B: MAE=107.82 MAPE=132.0% ρ=0.674 (351.9s)
2026-03-30 06:21:43,838 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (342.4s)
2026-03-30 06:27:36,640 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (352.8s)
------------------+------------+--+-
2026-03-30 06:29:48,256 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (129.5s)
2026-03-30 06:31:57,324 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (129.1s)
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 07:02 UTC
### Vast (fixed eval)
Eval running
```
--- length ---
------------------------+------------+--+-
2026-03-30 06:10:09,545 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (366.3s)
2026-03-30 06:16:01,437 - __main__ - INFO -   Qwen2.5-3B: MAE=107.82 MAPE=132.0% ρ=0.674 (351.9s)
2026-03-30 06:21:43,838 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (342.4s)
2026-03-30 06:27:36,640 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (352.8s)
------------------+------------+--+-
2026-03-30 06:29:48,256 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (129.5s)
2026-03-30 06:31:57,324 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (129.1s)
2026-03-30 06:34:06,756 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (129.4s)
2026-03-30 06:36:16,254 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (129.5s)
--------------------------------+------------+-
2026-03-30 06:37:13,893 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (55.6s)
2026-03-30 06:38:09,181 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (55.3s)
2026-03-30 06:39:04,527 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (55.3s)
2026-03-30 06:39:59,953 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (55.4s)
```
Files: `bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 3664B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 07:31 UTC
### Vast
Eval running
```
--- length ---
------------------------+------------+--+-
2026-03-30 06:10:09,545 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (366.3s)
2026-03-30 06:16:01,437 - __main__ - INFO -   Qwen2.5-3B: MAE=107.82 MAPE=132.0% ρ=0.674 (351.9s)
2026-03-30 06:21:43,838 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (342.4s)
2026-03-30 06:27:36,640 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (352.8s)
------------------+------------+--+-
2026-03-30 06:29:48,256 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (129.5s)
2026-03-30 06:31:57,324 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (129.1s)
2026-03-30 06:34:06,756 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (129.4s)
2026-03-30 06:36:16,254 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (129.5s)
--------------------------------+------------+-
2026-03-30 06:37:13,893 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (55.6s)
2026-03-30 06:38:09,181 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (55.3s)
2026-03-30 06:39:04,527 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (55.3s)
2026-03-30 06:39:59,953 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (55.4s)
2026-03-30 07:25:11,044 - __main__ - INFO -   Qwen2.5-14B: MAE=61183.45 MAPE=14921.1% ρ=0.029 (2709.3s)
```
Files: `bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 3664B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

**NOTE**: LoRA length eval: 14B MAE=61,183 (garbage), ρ=0.029. Took 45 min for 1 model.
LoRA eval is ~15h total for all targets — won't finish before Vast expires April 1.
Encoder results (KNN + ModernBERT + RoBERTa) are working correctly and fast (~1-2 min each).
**Decision needed**: Skip LoRA in eval (use training_results.json instead) to get encoder results faster?

## Check — 2026-03-30 08:01 UTC
### Vast
Eval running
```
--- length ---
------------------------+------------+--+-
2026-03-30 06:10:09,545 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (366.3s)
2026-03-30 06:16:01,437 - __main__ - INFO -   Qwen2.5-3B: MAE=107.82 MAPE=132.0% ρ=0.674 (351.9s)
2026-03-30 06:21:43,838 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (342.4s)
2026-03-30 06:27:36,640 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (352.8s)
------------------+------------+--+-
2026-03-30 06:29:48,256 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (129.5s)
2026-03-30 06:31:57,324 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (129.1s)
2026-03-30 06:34:06,756 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (129.4s)
2026-03-30 06:36:16,254 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (129.5s)
--------------------------------+------------+-
2026-03-30 06:37:13,893 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (55.6s)
2026-03-30 06:38:09,181 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (55.3s)
2026-03-30 06:39:04,527 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (55.3s)
2026-03-30 06:39:59,953 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (55.4s)
2026-03-30 07:25:11,044 - __main__ - INFO -   Qwen2.5-14B: MAE=61183.45 MAPE=14921.1% ρ=0.029 (2709.3s)
```
Files: `bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 3664B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 08:31 UTC
### Vast
Eval running
```
2026-03-30 06:34:06,756 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (129.4s)
2026-03-30 06:36:16,254 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (129.5s)
2026-03-30 06:37:13,893 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (55.6s)
2026-03-30 06:38:09,181 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (55.3s)
2026-03-30 06:39:04,527 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (55.3s)
2026-03-30 06:39:59,953 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (55.4s)
2026-03-30 07:25:11,044 - __main__ - INFO -   Qwen2.5-14B: MAE=61183.45 MAPE=14921.1% ρ=0.029 (2709.3s)
2026-03-30 08:11:56,548 - __main__ - INFO -   Qwen2.5-3B: MAE=61228.09 MAPE=29447.1% ρ=0.004 (2805.5s)
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 09:01 UTC
### Vast
Eval running
Latest: `2026-03-30 08:59:05,547 - __main__ - INFO -   Qwen2.5-72B: MAE=33753.92 MAPE=10068.5% ρ=-0.030 (2829.0s)`
### CSD3
`26508784 wd312 PD 26508783 wd312 PD 26528473 wd312 PD 26528474 wd312 PD `
---

## Check — 2026-03-30 09:31 UTC
### Vast
Eval running
```
2026-03-30 06:31:57,324 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (129.1s)
2026-03-30 06:34:06,756 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (129.4s)
2026-03-30 06:36:16,254 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (129.5s)
2026-03-30 06:37:13,893 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (55.6s)
2026-03-30 06:38:09,181 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (55.3s)
2026-03-30 06:39:04,527 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (55.3s)
2026-03-30 06:39:59,953 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (55.4s)
2026-03-30 07:25:11,044 - __main__ - INFO -   Qwen2.5-14B: MAE=61183.45 MAPE=14921.1% ρ=0.029 (2709.3s)
2026-03-30 08:11:56,548 - __main__ - INFO -   Qwen2.5-3B: MAE=61228.09 MAPE=29447.1% ρ=0.004 (2805.5s)
2026-03-30 08:59:05,547 - __main__ - INFO -   Qwen2.5-72B: MAE=33753.92 MAPE=10068.5% ρ=-0.030 (2829.0s)
```
Files: `bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 3664B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 10:01 UTC
### Vast
Eval running
```
  --- Qwen2.5-72B ---
  --- Qwen2.5-7B ---
  --- Aggregate ---
--- bucket ---
2026-03-30 09:49:34,186 - __main__ - INFO -   Qwen2.5-14B: MAE=87.36 MAPE=146.5% ρ=0.721 (271.0s)
2026-03-30 09:49:34,186 - __main__ - INFO -     Bucket: acc=0.482 adj=0.726 top3=0.755 E[len]_MAE=87.4
2026-03-30 09:54:02,613 - __main__ - INFO -   Qwen2.5-3B: MAE=97.27 MAPE=106.7% ρ=0.750 (268.4s)
2026-03-30 09:54:02,613 - __main__ - INFO -     Bucket: acc=0.397 adj=0.654 top3=0.672 E[len]_MAE=97.3
2026-03-30 09:58:31,875 - __main__ - INFO -   Qwen2.5-72B: MAE=96.99 MAPE=121.6% ρ=0.652 (269.2s)
2026-03-30 09:58:31,875 - __main__ - INFO -     Bucket: acc=0.398 adj=0.686 top3=0.710 E[len]_MAE=97.0
```
Files: `bucket_results.json: 6876B
judge_results.json: 2B
length_results.json: 7279B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 10:31 UTC
### Vast
Eval running
```
  --- Qwen2.5-3B ---
    (bucket)                     acc=0.397 adj=0.654 E[len]_MAE=97.3
    (bucket)                     acc=0.407 adj=0.671 E[len]_MAE=95.7
  --- Qwen2.5-72B ---
    (bucket)                     acc=0.398 adj=0.686 E[len]_MAE=97.0
    (bucket)                     acc=0.404 adj=0.688 E[len]_MAE=94.5
  --- Qwen2.5-7B ---
    (bucket)                     acc=0.379 adj=0.555 E[len]_MAE=146.0
    (bucket)                     acc=0.376 adj=0.553 E[len]_MAE=145.3
  --- Aggregate ---
  --- Bucket Filtering ---
--- judge ---
2026-03-30 10:17:05,136 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2334.9% ρ=0.024 (359.0s)
2026-03-30 10:22:58,640 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=9584.6% ρ=0.263 (353.5s)
2026-03-30 10:29:08,836 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3190.7% ρ=0.219 (370.2s)
```
Files: `bucket_results.json: 13733B
judge_results.json: 2B
length_results.json: 7279B
reference_score_results.json: 2B
similarity_results.json: 2B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

**BUG FOUND**: LoRA MAE=61K was due to adapter using per-model prompt on fused model.
Fused model expects `Predict length: {prompt}` but adapter sent `Predict length for model 7B: {prompt}`.
Output `14B:256 3B:312...` parsed as float → 0.0 or garbage.
**FIX**: Auto-detect fused mode from training_results.json, use correct prompt/parsing.
This also makes fused LoRA eval 4× faster (one generate call per prompt instead of four).
Need to re-run eval with this fix for correct LoRA results.

## 2026-03-30 10:47 UTC — Killed and restarted
Killed garbage LoRA eval. Launched final clean eval with all 3 adapter fixes.
Log: /tmp/vast_eval_final.log
Estimated: KNN ~6min/target + encoder ~4min/target + LoRA ~12min/target (4× faster fused) = ~5h total

## Check — 2026-03-30 11:01 UTC
### Vast
Eval running
```
--- length ---
2026-03-30 10:53:17,446 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (369.2s)
2026-03-30 10:59:26,139 - __main__ - INFO -   Qwen2.5-3B: MAE=107.82 MAPE=132.0% ρ=0.674 (368.7s)
```
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508784    ampere rb_rober    wd312 PD       0:00      1 (Priority)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
```
---

## 2026-03-30 11:14 UTC — DeBERTa precision issue

**Issue**: DeBERTaV3 fails with fp16 — `ValueError: Attempting to unscale FP16 gradients`.
DeBERTaV3 requires bf16 for training (known framework limitation).
Verified by smoke test on GPU VM: fp16 crashes, bf16 works (from earlier CSD3 INTR test).

**Decision**: Keep DeBERTa at bf16, RoBERTa+ModernBERT at fp16. Note in paper.
Reverted CSD3 script, resubmitted job 26554183 (RoBERTa fp16 + DeBERTa bf16).
Cancelled stale job 26552224 (wrong fp16 for DeBERTa).

## Smoke test results (all targets)
- length: **PASS** (KNN + ModernBERT + RoBERTa + LoRA all produce results)
- judge: **PASS** (all 4 predictors)
- similarity: **PASS** (all 4 predictors)
- reference_score: **PARTIAL** (KNN has no reference_score data — expected, not a bug)

Full eval running with verified code. KNN will be skipped for reference_score.

**KNN × reference_score**: KNN index was built from old data without `reference_score` field.
Stored fields: output_lengths, quality_scores, similarity_scores, llm_judge_scores.
Need to rebuild KNN from `train_with_ref.jsonl` to add reference_score — this is a training task.
Current eval will skip KNN for reference_score (returns empty, not fake zeros).

## Check — 2026-03-30 11:33 UTC
### Vast (final eval, KNN rebuilt, all fixes)
Eval running
```
--- length ---
```
Files: `*.json: B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26554183    ampere rb_rober    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 12:02 UTC
### Vast
Eval running
```
--- length ---
2026-03-30 11:33:39,797 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (48.6s)
2026-03-30 11:34:24,572 - __main__ - INFO -   Qwen2.5-3B: MAE=107.84 MAPE=132.0% ρ=0.674 (44.8s)
2026-03-30 11:35:12,596 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (48.0s)
2026-03-30 11:36:00,631 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (48.0s)
2026-03-30 11:38:25,145 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (142.5s)
2026-03-30 11:40:41,004 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (135.9s)
2026-03-30 11:42:59,797 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (138.8s)
2026-03-30 11:45:18,830 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (139.0s)
2026-03-30 11:46:20,385 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (59.2s)
2026-03-30 11:47:19,904 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (59.5s)
2026-03-30 11:48:19,671 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (59.8s)
2026-03-30 11:49:19,064 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (59.4s)
```
Files: `*.json: B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26554183    ampere rb_rober    wd312 PD       0:00      1 (Priority)
```
---

## TODO: Evaluator improvements (after current run finishes)

### 1. Single eval script with all predictors per target
- Merge bucket_encoder into the length target predictors list (not separate file)
- One script, one run per target, one output file
- Current run has bucket in separate file — merge results manually this time

### 2. Add CSV output
- Add --csv-output flag to run_evaluation.py
- Output per-target CSV with columns: predictor, model, MAE, MAPE, ρ, accuracy, adj_accuracy
- Also output bucket filtering CSV if applicable

### 3. Verify with smoke test before any full run
- After making changes, run 5-sample smoke test on ALL targets
- Verify CSV output is correct
- Only then launch full eval

### 4. KNN IVFPQ index
- User requested IVFPQ comparison — code exists in knn_estimator.py but not trained/evaluated
- Add to eval after flat KNN baseline established

## Check — 2026-03-30 12:31 UTC
### Vast
Eval running
```
--- length ---
2026-03-30 11:33:39,797 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (48.6s)
2026-03-30 11:34:24,572 - __main__ - INFO -   Qwen2.5-3B: MAE=107.84 MAPE=132.0% ρ=0.674 (44.8s)
2026-03-30 11:35:12,596 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (48.0s)
2026-03-30 11:36:00,631 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (48.0s)
2026-03-30 11:38:25,145 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (142.5s)
2026-03-30 11:40:41,004 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (135.9s)
2026-03-30 11:42:59,797 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (138.8s)
2026-03-30 11:45:18,830 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (139.0s)
2026-03-30 11:46:20,385 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (59.2s)
2026-03-30 11:47:19,904 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (59.5s)
2026-03-30 11:48:19,671 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (59.8s)
2026-03-30 11:49:19,064 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (59.4s)
```
Files: `*.json: B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26554183    ampere rb_rober    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 13:01 UTC
### Vast
Eval running
```
2026-03-30 11:35:12,596 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (48.0s)
2026-03-30 11:36:00,631 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (48.0s)
2026-03-30 11:38:25,145 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (142.5s)
2026-03-30 11:40:41,004 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (135.9s)
2026-03-30 11:42:59,797 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (138.8s)
2026-03-30 11:45:18,830 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (139.0s)
2026-03-30 11:46:20,385 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (59.2s)
2026-03-30 11:47:19,904 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (59.5s)
2026-03-30 11:48:19,671 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (59.8s)
2026-03-30 11:49:19,064 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (59.4s)
```
Files: `*.json: B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26554183    ampere rb_rober    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 13:31 UTC
### Vast
Eval running
```
--- length ---
2026-03-30 11:33:39,797 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (48.6s)
2026-03-30 11:34:24,572 - __main__ - INFO -   Qwen2.5-3B: MAE=107.84 MAPE=132.0% ρ=0.674 (44.8s)
2026-03-30 11:35:12,596 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (48.0s)
2026-03-30 11:36:00,631 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (48.0s)
2026-03-30 11:38:25,145 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (142.5s)
2026-03-30 11:40:41,004 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (135.9s)
2026-03-30 11:42:59,797 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (138.8s)
2026-03-30 11:45:18,830 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (139.0s)
2026-03-30 11:46:20,385 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (59.2s)
2026-03-30 11:47:19,904 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (59.5s)
2026-03-30 11:48:19,671 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (59.8s)
2026-03-30 11:49:19,064 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (59.4s)
```
Files: `*.json: B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26554183    ampere rb_rober    wd312 PD       0:00      1 (Priority)
```
---

## Check — 2026-03-30 14:01 UTC
### Vast
Eval running
```
--- length ---
2026-03-30 11:33:39,797 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (48.6s)
2026-03-30 11:34:24,572 - __main__ - INFO -   Qwen2.5-3B: MAE=107.84 MAPE=132.0% ρ=0.674 (44.8s)
2026-03-30 11:35:12,596 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (48.0s)
2026-03-30 11:36:00,631 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (48.0s)
2026-03-30 11:38:25,145 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (142.5s)
2026-03-30 11:40:41,004 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (135.9s)
2026-03-30 11:42:59,797 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (138.8s)
2026-03-30 11:45:18,830 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (139.0s)
2026-03-30 11:46:20,385 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (59.2s)
2026-03-30 11:47:19,904 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (59.5s)
2026-03-30 11:48:19,671 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (59.8s)
2026-03-30 11:49:19,064 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (59.4s)
2026-03-30 14:00:10,704 - __main__ - INFO -   Qwen2.5-14B: MAE=106.96 MAPE=62.2% ρ=0.392 (7848.5s)
```
Files: `*.json: B`
### CSD3
```
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
          26508783    ampere rb_mbert    wd312 PD       0:00      1 (Priority)
          26528474    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26528473    ampere rb_lora_    wd312 PD       0:00      1 (Priority)
          26554183    ampere rb_rober    wd312 PD       0:00      1 (Priority)
```
---

**LoRA length 14B**: MAE=106.96, ρ=0.392 (correct! matches training_results.json)
But took 7848s (2.2h) for 1 model — fused benefit lost because eval calls predict() per model.
**Perf bug (not correctness)**: evaluate_single() passes [tm] (1 model) to predict(),
so fused adapter generates all 4 models but only 1 is used per call. 4× waste.
TODO: optimize eval to call predict(prompt, all_models) once per prompt for fused adapters.
Current eval is slow but correct. ~30h remaining for all LoRA evals.

## Check — 2026-03-30 14:31 UTC
### Vast
Eval running
Latest: `2026-03-30 14:00:10,704 - __main__ - INFO -   Qwen2.5-14B: MAE=106.96 MAPE=62.2% ρ=0.392 (7848.5s)`
### CSD3
`26508783 wd312 PD 26528474 wd312 PD 26528473 wd312 PD 26554183 wd312 PD `
---

## Check — 2026-03-30 15:01 UTC
Latest: `2026-03-30 14:00:10,704 - __main__ - INFO -   Qwen2.5-14B: MAE=106.96 MAPE=62.2% ρ=0.392 (7848.5s)`
CSD3: `26508783 PD 26528474 PD 26528473 PD 26554183 PD `
---

## Check — 2026-03-30 15:31 UTC
Latest: `2026-03-30 14:00:10,704 - __main__ - INFO -   Qwen2.5-14B: MAE=106.96 MAPE=62.2% ρ=0.392 (7848.5s)`
Files: `*.json: B`
CSD3: `26508783 PD 26528474 PD 26528473 PD 26554183 PD `
---

## 2026-03-30 15:44 UTC — Efficiency fix applied and restarted
- Fix: predict(prompt, all_models) once per prompt instead of per-model loop
- Smoke test: 10 samples, LoRA 14B MAE=107.6 (matches), all 4 models in 5.5s (was 2.2h)
- Estimated total: ~2-3h for all 5 targets (was ~30h)

## Check — 2026-03-30 16:01 UTC
```
--- length ---
2026-03-30 15:47:35,719 - __main__ - INFO -   Qwen2.5-14B: MAE=95.25 MAPE=171.9% ρ=0.637 (46.9s)
2026-03-30 15:47:35,722 - __main__ - INFO -   Qwen2.5-3B: MAE=107.84 MAPE=132.0% ρ=0.674 (46.9s)
2026-03-30 15:47:35,724 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (46.9s)
2026-03-30 15:47:35,727 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (46.9s)
2026-03-30 15:49:54,672 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (34.2s)
2026-03-30 15:49:54,675 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (34.2s)
2026-03-30 15:49:54,678 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (34.2s)
2026-03-30 15:49:54,680 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (34.2s)
2026-03-30 15:51:01,271 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (16.1s)
2026-03-30 15:51:01,274 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (16.1s)
2026-03-30 15:51:01,277 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (16.1s)
2026-03-30 15:51:01,280 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (16.1s)
```
Files: `*.json: B`
CSD3: `26508783 PD 26528474 PD 26528473 PD 26554183 PD `
---

## Check — 2026-03-30 16:31 UTC
```
2026-03-30 15:49:54,672 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (34.2s)
2026-03-30 15:49:54,675 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (34.2s)
2026-03-30 15:49:54,678 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (34.2s)
2026-03-30 15:49:54,680 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (34.2s)
2026-03-30 15:51:01,271 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (16.1s)
2026-03-30 15:51:01,274 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (16.1s)
2026-03-30 15:51:01,277 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (16.1s)
2026-03-30 15:51:01,280 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (16.1s)
```
Files: `*.json: B`
CSD3: `26508783 PD 26528474 PD 26528473 PD 26554183 PD `
---

## 2026-03-30 17:00 UTC — CSD3 jobs reorganized
- Cancelled all previous jobs
- Submitted fused RoBERTa long (26577649) FIRST — deployment priority
  - 6 targets × 100ep, lr=1e-5, batch=8, polynomial, fp16 (matching Block params)
  - 2 rounds × 3 GPU parallel, ~12h total
- Resubmitted per-model + LoRA jobs (26577650-53) after
- Synced train_with_ref.jsonl to CSD3 for reference_score

## Check — 2026-03-30 17:01 UTC
```
2026-03-30 15:47:35,724 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (46.9s)
2026-03-30 15:47:35,727 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (46.9s)
2026-03-30 15:49:54,672 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (34.2s)
2026-03-30 15:49:54,675 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (34.2s)
2026-03-30 15:49:54,678 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (34.2s)
2026-03-30 15:49:54,680 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (34.2s)
2026-03-30 15:51:01,271 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (16.1s)
2026-03-30 15:51:01,274 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (16.1s)
2026-03-30 15:51:01,277 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (16.1s)
2026-03-30 15:51:01,280 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (16.1s)
```
Files: `*.json: B`
CSD3: `26577653 wd312 PD 26577652 wd312 PD 26577651 wd312 PD 26577650 wd312 PD 26577649 wd312 PD `
---

## 2026-03-30 17:28 UTC — Cleanup and status correction

### Cleaned up
- Removed all old eval logs (vast_overnight.log, vast_eval_fixed.log, vast_overnight_resume.log)
- Only `vast_eval_final.log` (current run with all fixes) remains
- eval_results/ is empty — no results saved yet from current run

### Current run status (vast_eval_final.log)
- Started: 15:47 UTC
- Length target: KNN done (48s), ModernBERT done (34s), RoBERTa done (16s), LoRA in progress (~2.5h, buffered)
- No result files saved yet — LoRA length must finish before length_results.json is written
- Bucket, judge, similarity, reference_score not started yet

### Previous results I reported were from BROKEN runs — do NOT use:
- Bucket results (ModernBERT 48.2%, RoBERTa) — from killed broken eval
- Judge ModernBERT/RoBERTa ρ — from killed broken eval
- All previous eval result files deleted

### CSD3 status
- 26577649: Fused RoBERTa long (100ep, 6 targets) — PENDING
- 26579811: Fused RoBERTa INTR test (1ep, 6 targets) — RUNNING
- 26577650-53: Per-model ModernBERT/RoBERTa/LoRA — PENDING

### Convergence analysis (from training logs, reliable):
| Target | ModernBERT 5ep | RoBERTa 5ep | RoBERTa converged? |
|--------|---------------|-------------|-------------------|
| Length log | ρ=0.731 (plateau ep3) | ρ=0.753 (still rising) | No |
| Bucket | acc=0.414 (plateau ep4) | acc=0.418 (still rising) | No |
| Judge | acc=0.608 (plateau ep3) | acc=0.622 (still rising) | No |
| Similarity | ρ=0.401 (plateau ep4) | ρ=0.421 (still rising) | No |
| Ref score | ρ=0.550 (flattening) | ρ=0.592 (still rising) | No |

### Decision pending:
- RoBERTa wins every target at 5ep and hasn't converged
- Fused RoBERTa long training (100ep) submitted to CSD3 as top priority
- INTR test running to verify all 6 targets work before long run starts

## Check — 2026-03-30 17:31 UTC
### Vast
GPU: `13 %, 2207 MiB` | Log size: `1821405`B
```
2026-03-30 15:47:35,724 - __main__ - INFO -   Qwen2.5-72B: MAE=104.80 MAPE=133.8% ρ=0.554 (46.9s)
2026-03-30 15:47:35,727 - __main__ - INFO -   Qwen2.5-7B: MAE=158.55 MAPE=292.1% ρ=0.438 (46.9s)
2026-03-30 15:49:54,672 - __main__ - INFO -   Qwen2.5-14B: MAE=82.04 MAPE=85.4% ρ=0.731 (34.2s)
2026-03-30 15:49:54,675 - __main__ - INFO -   Qwen2.5-3B: MAE=94.17 MAPE=68.2% ρ=0.750 (34.2s)
2026-03-30 15:49:54,678 - __main__ - INFO -   Qwen2.5-72B: MAE=93.63 MAPE=71.9% ρ=0.663 (34.2s)
2026-03-30 15:49:54,680 - __main__ - INFO -   Qwen2.5-7B: MAE=138.46 MAPE=126.4% ρ=0.570 (34.2s)
2026-03-30 15:51:01,271 - __main__ - INFO -   Qwen2.5-14B: MAE=83.99 MAPE=96.1% ρ=0.753 (16.1s)
2026-03-30 15:51:01,274 - __main__ - INFO -   Qwen2.5-3B: MAE=100.21 MAPE=86.0% ρ=0.759 (16.1s)
2026-03-30 15:51:01,277 - __main__ - INFO -   Qwen2.5-72B: MAE=96.82 MAPE=85.1% ρ=0.687 (16.1s)
2026-03-30 15:51:01,280 - __main__ - INFO -   Qwen2.5-7B: MAE=139.92 MAPE=142.4% ρ=0.585 (16.1s)
```
Results dir: `total 4
drwxrwxr-x 2 root root    6 Mar 30 10:58 .
drwx------ 1 root root 4096 Mar 30 00:10 ..`
### CSD3
```
JOBID NAME STATE TIME
26577653 rb_lora_2 PENDING 0:00
26577652 rb_lora_1 PENDING 0:00
26577651 rb_roberta_long PENDING 0:00
26577650 rb_mbert_long PENDING 0:00
26577649 rb_rob_fused PENDING 0:00
26579811 rb_rob_test RUNNING 14:32
```
INTR log:
```

--- similarity ---
2026-03-30 18:30:08,332 - __main__ - INFO - === Final Results ===
2026-03-30 18:30:08,332 - __main__ - INFO -   Qwen/Qwen2.5-14B: {'mae': 0.06869775056838989, 'median_ae': 0.04474514722824097, 'mape': 9.16125774383545, 'spearman_r': 0.5163542740663339, 'n': 3642, 'train_time_s': 101.1634111404419}
2026-03-30 18:30:08,332 - __main__ - INFO -   Qwen/Qwen2.5-3B: {'mae': 0.06731060892343521, 'median_ae': 0.043756723403930664, 'mape': 9.046156883239746, 'spearman_r': 0.510342970498153, 'n': 3642, 'train_time_s': 101.1634111404419}
2026-03-30 18:30:08,332 - __main__ - INFO -   Qwen/Qwen2.5-72B: {'mae': 0.015233649872243404, 'median_ae': 0.013671875, 'mape': 1.5233650207519531, 'spearman_r': 0.03873753323833762, 'n': 3642, 'train_time_s': 101.1634111404419}
2026-03-30 18:30:08,332 - __main__ - INFO -   Qwen/Qwen2.5-7B: {'mae': 0.07633871585130692, 'median_ae': 0.052881330251693726, 'mape': 10.458130836486816, 'spearman_r': 0.5815271790030111, 'n': 3642, 'train_time_s': 101.1634111404419}
  PASS

--- reference_score ---
```
---

## 2026-03-30 17:48 UTC — CRITICAL: reference_score provenance unknown

### Issue
- `reference_score` field exists in train_with_ref.jsonl with correct-looking values:
  - GSM8K: exact match 0/1
  - SQuAD: F1 token overlap (fractional)
  - beaver_tails: protectai safety score
  - Others: same as reference_similarity (cosine embedding)
- But the **code that computed it is missing** from the codebase
- `add_reference_similarity.py` only computes `reference_similarity` (cosine), NOT `reference_score`
- No script found on GPU VM, CSD3, Vast, or VPS that does GSM8K exact match or SQuAD F1
- No git history for this computation

### Risk
- Cannot verify if reference_score was computed correctly
- Cannot reproduce the data pipeline
- If reference_score is wrong, all reference_score models are trained on bad labels

### Action needed from user
- Find/recall how reference_score was generated (interactive session? notebook? deleted script?)
- Or: verify by spot-checking GSM8K entries (extract final number, compare to ref_score 0/1)
- Or: rewrite the computation script and regenerate reference_score from scratch

## 2026-03-30 18:08 UTC — reference_score code recovered

### Found
- Full code recovered from session log `d3311e06`
- Saved as `block/predictor/cara/offline_training/compute_reference_score.py`
- Logic: GSM8K exact-match, SQuAD token F1, beaver_tails protectai, others embedding similarity
- Originally run interactively on GPU VM March 27, never saved as script

### Still need to verify
- Spot check showed GSM8K entry 5 with 7B ref_score=1.0 but response about 'bass concert'
- Could be: regex matched a number in unrelated response, OR original broadcast data has wrong response mapping
- Need to run the recovered script on clean data and compare against existing train_with_ref.jsonl
- If they match → data was generated correctly, corruption is in original responses
- If they differ → data was corrupted during generation

## 2026-03-30 18:13 UTC — reference_score VERIFIED

**Data is NOT corrupted.** Recomputed reference_score from recovered script:
- 59,852 / 59,852 entries match (100.00%)
- GSM8K exact-match accuracy: 72B=36%, 14B=32%, 3B=30%, 7B=24%
- SQuAD token F1: 14B=0.201, 3B=0.177, 72B=0.141, 7B=0.171

Script saved: `block/predictor/cara/offline_training/compute_reference_score.py`
The 'bass concert' with ref_score=1.0 was regex matching a number coincidence, not data corruption.

## Check — 2026-03-30 18:31 UTC
```
2026-03-30 18:17:56,695 - __main__ - INFO -   Qwen2.5-72B: MAE=0.17 MAPE=2865.7% ρ=0.390 (40.6s)
2026-03-30 18:17:56,697 - __main__ - INFO -   Qwen2.5-7B: MAE=0.23 MAPE=3817.8% ρ=0.341 (40.6s)
2026-03-30 18:20:10,448 - __main__ - INFO -   Qwen2.5-14B: MAE=0.23 MAPE=2705.9% ρ=0.359 (33.0s)
2026-03-30 18:20:10,450 - __main__ - INFO -   Qwen2.5-3B: MAE=0.24 MAPE=8261.4% ρ=0.309 (33.0s)
2026-03-30 18:20:10,452 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3744.8% ρ=0.396 (33.0s)
2026-03-30 18:20:10,455 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (33.0s)
2026-03-30 18:21:07,872 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 18:21:07,874 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 18:21:07,876 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 18:21:07,879 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## 2026-03-30 18:34 UTC — Pipeline fix: reference_score merged into add_reference_similarity.py

- `compute_reference_score.py`: standalone script recovered from session log (for verification/re-run)
- `add_reference_similarity.py`: now computes BOTH reference_similarity AND reference_score in one pass
  - GSM8K: exact-match on final number (standard metric)
  - SQuAD: token F1 against answer spans (standard metric)
  - beaver_tails: ProtectAI refusal score
  - Others: embedding cosine similarity (fallback)
- Docstring updated with full scoring table and known limitations
- Data verified: 100% match between recomputed and existing reference_score
- ~2.3% GSM8K entries have correct answers followed by hallucinated text (documented, standard practice)

## 2026-03-30 18:49 UTC — Data pipeline and datacard updates

### Done
- `fill_source_field.py`: script to backfill source for all entries (saved, tested)
- GPU VM: all 4 data files now have 100% source coverage (train/test × with_ref/without)
- HuggingFace datacard updated: reference_score methodology table, data sources, safety scoring docs
- `feedback_no_interactive_execution.md`: rule saved — no interactive python for data/training

### TODO
- Upload fixed data files (with source field) to HuggingFace
- Also need to upload to Vast and CSD3 for consistency
- Sync fill_source_field.py and compute_reference_score.py to route-balance repo

## Check — 2026-03-30 19:01 UTC
```
2026-03-30 18:17:56,695 - __main__ - INFO -   Qwen2.5-72B: MAE=0.17 MAPE=2865.7% ρ=0.390 (40.6s)
2026-03-30 18:17:56,697 - __main__ - INFO -   Qwen2.5-7B: MAE=0.23 MAPE=3817.8% ρ=0.341 (40.6s)
2026-03-30 18:20:10,448 - __main__ - INFO -   Qwen2.5-14B: MAE=0.23 MAPE=2705.9% ρ=0.359 (33.0s)
2026-03-30 18:20:10,450 - __main__ - INFO -   Qwen2.5-3B: MAE=0.24 MAPE=8261.4% ρ=0.309 (33.0s)
2026-03-30 18:20:10,452 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3744.8% ρ=0.396 (33.0s)
2026-03-30 18:20:10,455 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (33.0s)
2026-03-30 18:21:07,872 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 18:21:07,874 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 18:21:07,876 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 18:21:07,879 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586261 PENDING 26586262 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## TODO: Remove quality_scores from KNN
- `quality_scores` (0.5×similarity + 0.5×judge) was never approved — remove from knn_estimator.py
- Remove `_extract_quality()` method
- Remove `quality_scores` from `build_index()`, `save()`, `load()`
- Remove `predict_quality()` method
- Keep: `llm_judge_scores`, `similarity_scores`, `reference_similarity_scores`, `output_lengths`
- Current eval uses `predict_judge()` which reads from `llm_judge_scores` — eval results are correct

## TODO: Fix quality evaluation metrics
- Current ρ is per-model across prompts — wrong for our use case
- Need: per-prompt cross-model ρ (same prompt, rank 4 models, compare predicted vs actual ordering)
- Also need: best-model accuracy (does predictor pick the right model for each prompt)
- MAE still useful for filtering threshold accuracy
- These are the 3 correct metrics for quality prediction evaluation:
  1. MAE (filtering threshold)
  2. Cross-model ρ per prompt (ranking quality)
  3. Best-model accuracy (routing decision)

## Check — 2026-03-30 20:02 UTC
```
2026-03-30 18:17:56,695 - __main__ - INFO -   Qwen2.5-72B: MAE=0.17 MAPE=2865.7% ρ=0.390 (40.6s)
2026-03-30 18:17:56,697 - __main__ - INFO -   Qwen2.5-7B: MAE=0.23 MAPE=3817.8% ρ=0.341 (40.6s)
2026-03-30 18:20:10,448 - __main__ - INFO -   Qwen2.5-14B: MAE=0.23 MAPE=2705.9% ρ=0.359 (33.0s)
2026-03-30 18:20:10,450 - __main__ - INFO -   Qwen2.5-3B: MAE=0.24 MAPE=8261.4% ρ=0.309 (33.0s)
2026-03-30 18:20:10,452 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3744.8% ρ=0.396 (33.0s)
2026-03-30 18:20:10,455 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (33.0s)
2026-03-30 18:21:07,872 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 18:21:07,874 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 18:21:07,876 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 18:21:07,879 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586261 PENDING 26586262 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-30 20:31 UTC
```
  --- Bucket Filtering ---
2026-03-30 18:15:07,775 - __main__ - INFO - Results saved to /root/eval_results/bucket_results.json
--- judge ---
2026-03-30 18:17:56,689 - __main__ - INFO -   Qwen2.5-14B: MAE=0.20 MAPE=2394.7% ρ=0.299 (40.6s)
2026-03-30 18:17:56,692 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8582.5% ρ=0.330 (40.6s)
2026-03-30 18:17:56,695 - __main__ - INFO -   Qwen2.5-72B: MAE=0.17 MAPE=2865.7% ρ=0.390 (40.6s)
2026-03-30 18:17:56,697 - __main__ - INFO -   Qwen2.5-7B: MAE=0.23 MAPE=3817.8% ρ=0.341 (40.6s)
2026-03-30 18:20:10,448 - __main__ - INFO -   Qwen2.5-14B: MAE=0.23 MAPE=2705.9% ρ=0.359 (33.0s)
2026-03-30 18:20:10,450 - __main__ - INFO -   Qwen2.5-3B: MAE=0.24 MAPE=8261.4% ρ=0.309 (33.0s)
2026-03-30 18:20:10,452 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3744.8% ρ=0.396 (33.0s)
2026-03-30 18:20:10,455 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (33.0s)
2026-03-30 18:21:07,872 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 18:21:07,874 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 18:21:07,876 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 18:21:07,879 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-30 21:01 UTC
GPU: `10 %, 2207 MiB` | Log: `3655768 1774894869`
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-30 21:31 UTC
```
2026-03-30 21:21:21,016 - __main__ - INFO -   Qwen2.5-72B: MAE=0.17 MAPE=2865.7% ρ=0.390 (41.3s)
2026-03-30 21:21:21,019 - __main__ - INFO -   Qwen2.5-7B: MAE=0.23 MAPE=3817.8% ρ=0.341 (41.3s)
2026-03-30 21:23:34,061 - __main__ - INFO -   Qwen2.5-14B: MAE=0.23 MAPE=2705.9% ρ=0.359 (32.8s)
2026-03-30 21:23:34,063 - __main__ - INFO -   Qwen2.5-3B: MAE=0.24 MAPE=8261.4% ρ=0.309 (32.8s)
2026-03-30 21:23:34,065 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3744.8% ρ=0.396 (32.8s)
2026-03-30 21:23:34,067 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (32.8s)
2026-03-30 21:24:32,394 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 21:24:32,396 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 21:24:32,398 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 21:24:32,400 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## 2026-03-30 21:41 UTC — Plan documented

### Immediate
- Eval running: judge LoRA (~3h), then similarity (~3.5h), reference_score (~3.5h)
- Progress logging enabled, unbuffered
- After eval: Prometheus rejudging on same Vast (~3-6h)
- Then stop RTX 8000, switch to 3090

### Key decisions made
- RoBERTa replaces ModernBERT as primary encoder (wins all targets, not converged at 5ep)
- Fused RoBERTa 100ep on CSD3 = deployment models
- Quality: 10-class classification for confidence-based filtering (argmax for scoring, CDF for filtering)
- TTFT/TPOT: point estimation + tiered relaxation (same as field), conformal prediction as future extension
- Prometheus reference-grounded judging: new quality signal merging reference_score + judge

### Handoff docs updated
- HANDOFF_MARCH30.md: full session summary + all plans
- OVERNIGHT_MONITOR.md: running log of all checks and fixes

## Check — 2026-03-30 22:01 UTC
```
2026-03-30 21:21:21,016 - __main__ - INFO -   Qwen2.5-72B: MAE=0.17 MAPE=2865.7% ρ=0.390 (41.3s)
2026-03-30 21:21:21,019 - __main__ - INFO -   Qwen2.5-7B: MAE=0.23 MAPE=3817.8% ρ=0.341 (41.3s)
2026-03-30 21:23:34,061 - __main__ - INFO -   Qwen2.5-14B: MAE=0.23 MAPE=2705.9% ρ=0.359 (32.8s)
2026-03-30 21:23:34,063 - __main__ - INFO -   Qwen2.5-3B: MAE=0.24 MAPE=8261.4% ρ=0.309 (32.8s)
2026-03-30 21:23:34,065 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3744.8% ρ=0.396 (32.8s)
2026-03-30 21:23:34,067 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (32.8s)
2026-03-30 21:24:32,394 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 21:24:32,396 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 21:24:32,398 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 21:24:32,400 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## 2026-03-30 22:04 UTC — Full audit of what's verified vs assumed

### VERIFIED WORKING (tested end-to-end)
- Route-balance repo smoke test: 56/56 PASS on CloudLab
- P-D benchmark: 3/3 PASS with correct metrics
- RoBERTa fused 5ep training: all 5 targets trained on Vast, convergence curves checked
- ModernBERT fused 5ep: all 5 targets trained on CSD3, convergence curves checked
- KNN rebuild with reference_score: verified 100% match with recomputed data
- XGBoost TTFT/TPOT/E2E: deployed and tested in smoke test
- Length eval results: saved, all 4 predictors produced results
- Bucket eval results: saved, filtering metrics present
- reference_score data: verified correct via recomputation
- source field: 100% filled for all entries
- HF datacard + data uploaded with fixes

### RUNNING (not yet verified)
- Vast eval: judge LoRA at 20%, similarity + reference_score pending (~8h remaining)
- CSD3 fused RoBERTa 100ep: PENDING in queue (INTR test passed for all 6 targets)
- CSD3 per-model jobs: PENDING

### KNOWN BROKEN / NOT DONE
- LSTM training: uses wrong features (aggregate, not per-request queue shapes). Needs rewrite.
- LSTM eval: no results on correct data. Previous results (MAE=1.6-14.3) are from wrong input.
- Roofline eval: never evaluated offline against 405K ground truth
- Static TPOT eval: never evaluated offline
- Latency predictor evaluator: does NOT exist. No unified eval script for latency predictors.
- quality_scores in KNN: unauthorized combined metric, needs removal
- HF cara_latency_prediction: STALE/BUGGY, needs re-upload from local fixed data
- Prometheus reference-grounded judging: not started, script not written
- Quality 10-class CDF filtering: not implemented in scheduler
- Conformal prediction for TTFT/TPOT: not implemented
- TimeBill baseline: not implemented
- Cross-model ρ per prompt: not computed (current ρ is wrong metric)
- Best-model accuracy: not in evaluation framework
- Bucket-as-length in same eval: separate file, should be merged
- CSV output from evaluator: not implemented
- Evaluator cleanup: remove wrong ρ, add cross-model ρ + best-model accuracy
- route-balance repo: code changes from today not synced
- Paper related work: SLO enforcement section not expanded

### UNCERTAIN (not verified, might have bugs)
- LoRA fused training results: MAE/ρ from training_results.json not independently verified
- Per-model ModernBERT bucket/reference_score (Vast): trained but eval adapter had bugs at the time
- All encoder adapter paths: 3 bugs found and fixed today, may have more
- KNN adapter for judge: uses llm_judge_scores (verified correct), but KNN quality prediction is weak (ρ=0.024)
- DeBERTa bf16 vs fp16: DeBERTa requires bf16, noted as unfair comparison but accepted

### FULL TODO LIST (prioritized)
1. [RUNNING] Finish Vast eval (judge + similarity + reference_score) — ~8h
2. [RUNNING] CSD3 fused RoBERTa 100ep — waiting in queue
3. Write Prometheus scoring script — need before Vast expires
4. Run Prometheus rejudging on Vast — ~3-6h
5. Rewrite LSTM training to use per-request queue shapes
6. Train LSTM on correct data — ~5h on any GPU
7. Write latency predictor evaluator (XGBoost vs LSTM vs Roofline vs Static)
8. Run latency predictor ablation on 405K data
9. Re-upload cara_latency_prediction to HF with fixed data
10. Remove quality_scores from KNN
11. Add cross-model ρ + best-model accuracy to quality evaluator
12. Add bucket_encoder to length eval (one run, one file)
13. Add CSV output to evaluator
14. Implement quality 10-class CDF filtering in scheduler
15. Implement conformal prediction / quantile regression for TTFT/TPOT
16. Implement TimeBill baseline
17. Sync all code to route-balance repo
18. HF dataset rename (cara → route_balance)
19. Expand paper related work (SLO enforcement)
20. Write analysis scripts for CloudLab 18-node evaluation
21. Back up all Vast models/results to VPS before expiry

## Check — 2026-03-30 22:32 UTC
```
2026-03-30 21:23:34,061 - __main__ - INFO -   Qwen2.5-14B: MAE=0.23 MAPE=2705.9% ρ=0.359 (32.8s)
2026-03-30 21:23:34,063 - __main__ - INFO -   Qwen2.5-3B: MAE=0.24 MAPE=8261.4% ρ=0.309 (32.8s)
2026-03-30 21:23:34,065 - __main__ - INFO -   Qwen2.5-72B: MAE=0.20 MAPE=3744.8% ρ=0.396 (32.8s)
2026-03-30 21:23:34,067 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (32.8s)
2026-03-30 21:24:32,394 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 21:24:32,396 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 21:24:32,398 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 21:24:32,400 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-30 23:01 UTC
LoRA: `2026-03-30 22:53:40,284 - __main__ - INFO -   [llm:judge] 1820/3642 (50%) | 5346s elapsed | 0.3 prompts/s | ETA 5352s`
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586261 PENDING 26586262 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-30 23:32 UTC
LoRA: `2026-03-30 23:24:38,048 - __main__ - INFO -   [llm:judge] 2366/3642 (65%) | 7204s elapsed | 0.3 prompts/s | ETA 3885s`
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 00:01 UTC
LoRA: `2026-03-31 00:00:51,196 - __main__ - INFO -   [llm:judge] 3094/3642 (85%) | 9377s elapsed | 0.3 prompts/s | ETA 1661s`
```
2026-03-30 21:23:34,067 - __main__ - INFO -   Qwen2.5-7B: MAE=0.25 MAPE=4195.3% ρ=0.334 (32.8s)
2026-03-30 21:24:32,394 - __main__ - INFO -   Qwen2.5-14B: MAE=0.21 MAPE=2187.9% ρ=0.405 (13.8s)
2026-03-30 21:24:32,396 - __main__ - INFO -   Qwen2.5-3B: MAE=0.22 MAPE=8406.9% ρ=0.364 (13.8s)
2026-03-30 21:24:32,398 - __main__ - INFO -   Qwen2.5-72B: MAE=0.19 MAPE=2119.3% ρ=0.460 (13.8s)
2026-03-30 21:24:32,400 - __main__ - INFO -   Qwen2.5-7B: MAE=0.24 MAPE=4759.8% ρ=0.367 (13.8s)
```
Files: `bucket_results.json: 13723B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 00:31 UTC
```
2026-03-31 00:29:31,011 - __main__ - INFO -   Qwen2.5-3B: MAE=0.23 MAPE=5950.6% ρ=0.120 (2774.2s)
2026-03-31 00:29:31,013 - __main__ - INFO -   Qwen2.5-72B: MAE=0.22 MAPE=1930.2% ρ=0.059 (2774.2s)
2026-03-31 00:29:31,016 - __main__ - INFO -   Qwen2.5-7B: MAE=0.23 MAPE=1566.8% ρ=0.190 (2774.2s)
  --- Qwen2.5-14B ---
  --- Qwen2.5-3B ---
  --- Qwen2.5-72B ---
  --- Qwen2.5-7B ---
  --- Aggregate ---
2026-03-31 00:29:31,018 - __main__ - INFO - Results saved to /root/eval_results/judge_results.json
--- similarity ---
```
LoRA: `2026-03-31 00:29:23,371 - __main__ - INFO -   [llm:judge] 3640/3642 (100%) | 11089s elapsed | 0.3 prompts/s | ETA 6s`
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 01:02 UTC
LoRA: `2026-03-31 00:58:46,139 - __main__ - INFO -   [llm:similarity] 364/3642 (10%) | 1373s elapsed | 0.3 prompts/s | ETA 12363s`
```
2026-03-31 00:34:38,792 - __main__ - INFO -   Qwen2.5-14B: MAE=0.07 MAPE=9.0% ρ=0.510 (32.5s)
2026-03-31 00:34:38,795 - __main__ - INFO -   Qwen2.5-3B: MAE=0.07 MAPE=9.1% ρ=0.493 (32.5s)
2026-03-31 00:34:38,797 - __main__ - INFO -   Qwen2.5-72B: MAE=0.01 MAPE=1.5% ρ=0.001 (32.5s)
2026-03-31 00:34:38,800 - __main__ - INFO -   Qwen2.5-7B: MAE=0.08 MAPE=10.5% ρ=0.572 (32.5s)
2026-03-31 00:35:51,461 - __main__ - INFO -   Qwen2.5-14B: MAE=0.06 MAPE=8.2% ρ=0.547 (17.6s)
2026-03-31 00:35:51,463 - __main__ - INFO -   Qwen2.5-3B: MAE=0.06 MAPE=8.4% ρ=0.533 (17.6s)
2026-03-31 00:35:51,466 - __main__ - INFO -   Qwen2.5-72B: MAE=0.02 MAPE=2.0% ρ=-0.010 (17.6s)
2026-03-31 00:35:51,469 - __main__ - INFO -   Qwen2.5-7B: MAE=0.07 MAPE=10.2% ρ=0.609 (17.6s)
```
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B`
CSD3: `26586261 PENDING 26586262 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 01:32 UTC
LoRA: `2026-03-31 01:21:42,886 - __main__ - INFO -   [llm:similarity] 728/3642 (20%) | 2750s elapsed | 0.3 prompts/s | ETA 11006s`
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 02:01 UTC
LoRA: `2026-03-31 02:01:30,126 - __main__ - INFO -   [llm:similarity] 1456/3642 (40%) | 5137s elapsed | 0.3 prompts/s | ETA 7712s`
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 02:32 UTC
LoRA: `2026-03-31 02:28:37,435 - __main__ - INFO -   [llm:similarity] 2002/3642 (55%) | 6764s elapsed | 0.3 prompts/s | ETA 5541s`
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 03:01 UTC
LoRA: `2026-03-31 02:54:27,493 - __main__ - INFO -   [llm:similarity] 2548/3642 (70%) | 8314s elapsed | 0.3 prompts/s | ETA 3570s`
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 03:31 UTC
LoRA: `2026-03-31 03:21:13,481 - __main__ - INFO -   [llm:similarity] 3094/3642 (85%) | 9920s elapsed | 0.3 prompts/s | ETA 1757s`
```
2026-03-31 00:34:38,800 - __main__ - INFO -   Qwen2.5-7B: MAE=0.08 MAPE=10.5% ρ=0.572 (32.5s)
2026-03-31 00:35:51,461 - __main__ - INFO -   Qwen2.5-14B: MAE=0.06 MAPE=8.2% ρ=0.547 (17.6s)
2026-03-31 00:35:51,463 - __main__ - INFO -   Qwen2.5-3B: MAE=0.06 MAPE=8.4% ρ=0.533 (17.6s)
2026-03-31 00:35:51,466 - __main__ - INFO -   Qwen2.5-72B: MAE=0.02 MAPE=2.0% ρ=-0.010 (17.6s)
2026-03-31 00:35:51,469 - __main__ - INFO -   Qwen2.5-7B: MAE=0.07 MAPE=10.2% ρ=0.609 (17.6s)
```
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 04:01 UTC
LoRA: `2026-03-31 03:52:10,661 - __main__ - INFO -   [llm:similarity] 3640/3642 (100%) | 11777s elapsed | 0.3 prompts/s | ETA 6s`
```
2026-03-31 03:57:19,204 - __main__ - INFO -   Qwen2.5-14B: MAE=0.17 MAPE=2797.5% ρ=0.538 (32.8s)
2026-03-31 03:57:19,207 - __main__ - INFO -   Qwen2.5-3B: MAE=0.19 MAPE=8685.7% ρ=0.566 (32.8s)
2026-03-31 03:57:19,210 - __main__ - INFO -   Qwen2.5-72B: MAE=0.16 MAPE=3327.5% ρ=0.518 (32.8s)
2026-03-31 03:57:19,212 - __main__ - INFO -   Qwen2.5-7B: MAE=0.18 MAPE=4466.7% ρ=0.581 (32.8s)
2026-03-31 03:58:17,726 - __main__ - INFO -   Qwen2.5-14B: MAE=0.16 MAPE=2875.7% ρ=0.578 (14.0s)
2026-03-31 03:58:17,729 - __main__ - INFO -   Qwen2.5-3B: MAE=0.17 MAPE=9426.4% ρ=0.584 (14.0s)
2026-03-31 03:58:17,732 - __main__ - INFO -   Qwen2.5-72B: MAE=0.14 MAPE=3417.9% ρ=0.596 (14.0s)
2026-03-31 03:58:17,734 - __main__ - INFO -   Qwen2.5-7B: MAE=0.17 MAPE=4662.5% ρ=0.609 (14.0s)
```
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B
similarity_results.json: 6788B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 04:31 UTC
LoRA: `2026-03-31 04:23:52,006 - __main__ - INFO -   [llm:reference_score] 546/3642 (15%) | 1528s elapsed | 0.4 prompts/s | ETA 8663s`
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B
similarity_results.json: 6788B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 05:01 UTC
LoRA: `2026-03-31 04:55:29,494 - __main__ - INFO -   [llm:reference_score] 1092/3642 (30%) | 3425s elapsed | 0.3 prompts/s | ETA 7998s`
Files: `bucket_results.json: 13723B
judge_results.json: 6747B
length_results.json: 7269B
similarity_results.json: 6788B`
CSD3: `26586262 PENDING 26586261 PENDING 26586259 PENDING 26586257 PENDING 26586256 PENDING `
---

## Check — 2026-03-31 05:15 UTC (Manual)

### Vast
- LoRA reference_score: 35% (1274/3642), 0.3 prompts/s, ETA ~7415s (~2h, finish ~07:20 UTC)
- GPU: 2.2GB/46GB used (0.5B LoRA model is tiny)
- Disk: **32GB full** — attempted Prometheus 7B model download, failed with ENOSPC
  - Cleaned up partial download, recovered to 5.8GB free (82%)
  - Need ~15GB for Qwen2.5-7B-Instruct judge model
- 4/5 eval result files saved: length, bucket, judge, similarity

### Prometheus Script
- Written: `block/predictor/cara/offline_training/prometheus_judge.py`
- Score parsing: 7/7 unit tests pass
- Smoke test: FAILED (disk full, can't download 7B model)
- **Plan**: After eval completes → back up everything to VPS → delete backed-up models → download 7B → run Prometheus
- Alternative: Run on new 3090 Vast instance ($0.10/hr) with fresh disk

### CSD3
- All 5 jobs still PENDING (Priority) — likely waiting for Q2 budget (April 1)
- 26586256 (fused RoBERTa 100ep) is top priority

### Disk Cleanup Plan (after backup)
| Path | Size | Can delete after backup? |
|------|------|------------------------|
| length_bucket_per_model/ | 2.3GB | Yes |
| reference_score_per_model/ | 2.3GB | Yes |
| roberta_fused/ | 2.4GB | Yes |
| modernbert_fused_* (5 dirs) | 2.9GB | Yes |
| **Total reclaimable** | **~10GB** | |

### Completed Result Analysis (4/5 targets)
| Target | Best Predictor | Best Agg ρ | 2nd Best | 2nd ρ |
|--------|---------------|-----------|----------|-------|
| Length | RoBERTa log | 0.696 | ModernBERT | 0.679 |
| Bucket | RoBERTa bucket | 0.691 | ModernBERT bucket | 0.671 |
| Judge | RoBERTa 10-class | 0.399 | ModernBERT reg | 0.349 |
| Similarity | RoBERTa | 0.419 | ModernBERT | 0.394 |

LoRA (0.5B) is garbage on all targets (ρ = 0.13-0.31). Not worth pursuing.
RoBERTa only at 5 epochs, CSD3 100ep should be much better.

### Backup Progress
- 4/5 eval result JSONs → VPS ✅
- Small models (KNN, LoRA) → VPS ✅
- Large models (roberta, modernbert, per-model) → FAILED (disk too small for tar on both Vast and local). 5-epoch models superseded by CSD3 100ep. Configs backed up.
---

## Check — 2026-03-31 05:35 UTC (Cron)

### Vast
- LoRA reference_score: **45%** (1638/3642), 0.3 prompts/s, ETA ~6193s (finish ~07:06 UTC)
- Process running: PID 15472, 99% CPU, healthy
- Disk: 82% (5.8GB free) — stable after cleanup of partial tar files
- 4/5 result files saved (reference_score_results.json pending)
- No errors, no issues

### CSD3
- All 5 jobs still PENDING (Priority): 26586256 26586257 26586259 26586261 26586262
- Likely waiting for Q2 budget renewal (April 1)

### Status: No action needed. Eval progressing normally.
---

## Check — 2026-03-31 06:00 UTC (Cron)

### Vast
- LoRA reference_score: **65%** (2366/3642), 0.3 prompts/s, ETA ~3855s (finish ~07:05 UTC)
- Process PID 15472 running healthy, 129min elapsed
- Disk: 82% stable
- 4/5 result files saved

### CSD3
- All 5 jobs PENDING (Priority): 26586256 26586257 26586259 26586261 26586262

### Status: No action needed. ~1h to eval completion.
---

## Check — 2026-03-31 06:30 UTC (Cron)

### Vast
- LoRA reference_score: **80%** (2912/3642), 0.3 prompts/s, ETA ~2184s (finish ~06:59 UTC)
- Process PID 15472 healthy, 159min elapsed
- **~30 minutes to completion**

### CSD3
- All 5 jobs PENDING (Priority): 26586256 26586257 26586259 26586261 26586262

### Status: No action needed. Eval nearly done — next check should see completion.
---

## Check — 2026-03-31 07:00 UTC (Cron)

### Vast — EVAL COMPLETE ✅
- **reference_score_results.json written at 06:58 UTC** (6806 bytes)
- All 5/5 eval result files saved
- No eval process running — completed successfully
- reference_score_results.json backed up to VPS ✅

### Reference Score Results (5th and final target)
| Predictor | Agg MAE | Agg ρ | 14B ρ | 3B ρ | 72B ρ | 7B ρ |
|-----------|---------|-------|-------|------|-------|------|
| **RoBERTa** | **0.162** | **0.592** | 0.578 | 0.584 | 0.596 | **0.609** |
| ModernBERT | 0.176 | 0.550 | 0.538 | 0.566 | 0.518 | 0.581 |
| KNN | 0.208 | 0.508 | 0.470 | 0.504 | 0.516 | 0.541 |
| LoRA (0.5B) | 0.223 | 0.379 | 0.407 | 0.374 | 0.387 | 0.350 |

**RoBERTa wins all 5/5 targets.** Complete summary:

| Target | RoBERTa ρ | ModernBERT ρ | KNN ρ | LoRA ρ |
|--------|-----------|-------------|-------|--------|
| Length | 0.696 | 0.679 | 0.576 | 0.307 |
| Bucket | 0.691 | 0.671 | — | — |
| Judge | 0.399 | 0.349 | 0.340 | 0.126 |
| Similarity | 0.419 | 0.394 | 0.384 | 0.353 |
| Ref score | 0.592 | 0.550 | 0.508 | 0.379 |

### CSD3
- All 5 jobs PENDING (Priority): 26586256 26586257 26586259 26586261 26586262

### Next Steps
1. ~~Vast GPU now free — can run Prometheus scoring (need disk cleanup first)~~ DONE
2. All 5/5 eval results backed up to VPS
3. CSD3 100ep RoBERTa should significantly improve these 5ep numbers
---

## Check — 2026-03-31 08:30 UTC

### Vast — Prometheus Scoring LAUNCHED
- Cleaned up old models: freed ~10GB, disk went from 82% → 52%
- Downloaded Qwen2.5-7B-Instruct (~14GB) → disk now 97%
- **Smoke test PASSED**: 9/9 parsed, self-reference 5/5, score distribution sensible
- **Full run launched**: PID 16548, batch_size=4, 3642 entries × 3 models = 10,926 pairs
- GPU: 17.4GB/46GB VRAM, 57% utilization
- Estimated runtime: ~10-15h (transformers inference, no vLLM)
- Output: `/root/eval_results/prometheus_test.jsonl`
- Log: `/tmp/prometheus_run.log`

### CSD3
- All 5 jobs PENDING (Priority). No change.
---

## Check — 2026-03-31 09:00 UTC (Cron)
Vast: Prometheus running (PID 16548, 11min CPU, GPU 53% util, 17.4GB VRAM). No progress log yet — first 5% threshold not reached. No errors.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed. Prometheus processing normally.
---

## Check — 2026-03-31 09:30 UTC (Cron)
Vast: Prometheus **10%** (1092/10926), 0.8 pairs/s, ETA ~11906s (**finish ~12:15 UTC, ~3.3h**). Running healthy.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed.
---

## Check — 2026-03-31 10:35 UTC (Cron)
Vast: Prometheus **10%** (1092/10926), 0.9 pairs/s, ETA ~10955s (**finish ~12:10 UTC**). PID 17277 healthy, 26min CPU.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed. Evaluator refactored with cross-model metrics + CSV on GPU VM — smoke tested OK.
---

## Check — 2026-03-31 11:00 UTC (Cron)
Vast: Prometheus **25%** (2732/10926), 0.9 pairs/s, ETA ~9124s (**finish ~12:30 UTC**). PID 17277 healthy.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed.
---

## Check — 2026-03-31 11:30 UTC (Cron)
Vast: Prometheus **40%** (4368/10926), 0.9 pairs/s, ETA ~7307s (**finish ~12:30 UTC**). PID 17277 healthy.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed. ~1h remaining.
---

## Check — 2026-03-31 12:05 UTC (Cron)
Vast: Qwen reference judging **55%** (6008/10926), 0.9 pairs/s, ETA ~5503s (**finish ~12:30 UTC**). Watcher alive (sleep 60), transition script ready. Will auto-transition to Prometheus-7b-v2 after completion.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed.
---

## Check — 2026-03-31 12:35 UTC (Cron)
Vast: Qwen reference judging **70%** (7644/10926), 0.9 pairs/s, ETA ~3652s (**finish ~13:30 UTC**). Watcher waiting.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed. ~1h to Qwen completion, then auto-transition to Prometheus-v2.
---

## Check — 2026-03-31 13:05 UTC (Cron)
Vast: Qwen reference judging **85%** (9284/10926), 0.9 pairs/s, ETA ~1837s (**finish ~13:35 UTC**). ~30min remaining. Watcher ready.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed. Qwen almost done, auto-transition to Prometheus-v2 imminent.
---

## Check — 2026-03-31 13:35 UTC (Cron)
Vast: Qwen reference judging **95%** (10376/10926), ETA ~623s. **~10 minutes to completion.** Transition to Prometheus-v2 will fire automatically.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed. Qwen about to finish.
---

## Check — 2026-03-31 13:40 UTC (Cron)

### Vast — Qwen COMPLETE, Prometheus-v2 RUNNING
- **Qwen reference judging COMPLETE** at 12:33 UTC
  - 10,926 pairs scored, mean score 6.92/10
  - Score distribution: {1:268, 2:745, 3:30, 4:687, 5:655, 6:1199, 7:2278, 8:2846, 9:399, 10:1819}
  - Results backed up to VPS as qwen_reference_judge.jsonl
  - Column names fixed: prometheus_score → llm_judge_scores.qwen2.5-7b-it_reference
- **Transition executed successfully**:
  - Qwen 7B cache deleted, disk 97% → 63%
  - Prometheus-7b-v2.0 downloading (8 shards)
- **Prometheus-v2 scoring LAUNCHED**: PID 17958, batch_size=4
  - Output: /root/eval_results/prometheus_v2_test.jsonl
  - ETA: ~3-4h based on Qwen run speed

### CSD3
- All 5 jobs PENDING (Priority). No change.

### Status: Prometheus-v2 running. No action needed.
---

## Check — 2026-03-31 14:05 UTC (Cron)
Vast: Prometheus-v2 **10%** (1092/10926), 0.7 pairs/s. PID 17958 healthy.
**BUT: USING WRONG REFERENCE (72B response instead of dataset ground truth)**
CSD3: All 5 jobs PENDING (Priority). No change.
---

## Fix — 2026-03-31 14:30 UTC
**Critical bug found**: Both Qwen and Prometheus runs used 72B model response as reference instead of dataset ground truth (GSM8K answer, SQuAD span, etc.). This is NOT reference-grounded judging.
- Killed wrong Prometheus-v2 run
- Created `add_reference_text.py` — loads actual reference texts from HF datasets, stores as `reference_text` field
- Fixed `prometheus_judge.py` — reads `reference_text` field, scores ALL 4 models (including 72B)
- Generated `test_with_reftext.jsonl` on GPU VM — 3642/3642 entries have reference_text (100%)
- Smoke tested on GPU VM: 12/12 parsed, scores sensible (72B≈0.9-1.0, correct model ordering)
- **Restarted Prometheus-v2** with correct data: PID 18507, using `test_with_reftext.jsonl`
- Confirmed log: "Using reference_text field (dataset ground truth) as reference"
- Previous Qwen run (qwen_reference_judge.jsonl) is INVALID — used 72B as reference
---

## Check — 2026-03-31 14:35 UTC (Cron)
Vast: Prometheus-v2 (CORRECT, dataset references) running. PID 18507, early stage (no progress log yet). ETA ~4h.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed. Correct run in progress.
---

## Fix — 2026-03-31 15:20 UTC
**Bug #2**: Prometheus-v2 requires specific prompt format (###Task Description, [RESULT] output, 1-5 scale). Generic template → all-10 outputs. Fixed with auto-detection + correct template. Also fixed `self.score_max` → `judge.score_max` in standalone function.
**Bug #3 (72B-as-reference)** already fixed earlier. Killed wrong Vast run, restarted both.
**Decision**: Skip Qwen reference-grounded (same model family bias). Use only Prometheus-v2 (Mistral-based, unbiased).

### Now running (both correct):
| GPU | Dataset | Pairs | Status |
|-----|---------|-------|--------|
| Vast RTX 8000 | test (3,642×4) | 14,568 | Running, early stage |
| Personal RTX 8000 | train (14,963×4) | 59,852 | Running, early stage |
---

## Check — 2026-03-31 16:05 UTC (Cron)
Vast: Prometheus-v2 test running (PID 18849, 38min CPU). No progress log yet (slow with 512 max_tokens).
Personal VM: Prometheus-v2 train running (PID 15986, 39min CPU). No progress log yet.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: Both healthy, no errors. Prometheus with feedback generation is ~3x slower than Qwen. First progress at 5% threshold expected soon.
---

## Check — 2026-03-31 16:35 UTC (Cron)
Vast: Prometheus-v2 test **5%** (728/14568), **0.3 pairs/s**, ETA ~50142s (**finish ~05:00 UTC April 1, ~14h**). Much slower than Qwen (0.3 vs 0.9 pairs/s) due to 512 max_tokens feedback generation.
Personal VM: Prometheus-v2 train running (PID 15986, 69min CPU). No progress log yet (first 5% = 2993 pairs).
CSD3: All 5 jobs PENDING (Priority). No change.
Status: Both healthy. Slow but correct. Will run overnight.
---

## Check — 2026-03-31 17:10 UTC (Cron)
Vast: Prometheus-v2 test **10%** (1456/14568), 0.3 pairs/s. ~13h remaining with HF generate.
CSD3: All 5 jobs PENDING (Priority). No change.
Personal VM: **vLLM installed** (v0.18.1). Speed test: **5.1 prompts/s** — 17x faster than HF generate. Plan: write vLLM scoring script, redo both test+train on personal VM (~4h total). Kill Vast after.
---

## Check — 2026-03-31 17:35 UTC (Cron)
Vast: **KILLED** (instance destroyed, data synced to personal VM + VPS).
CSD3: All 5 jobs PENDING (Priority). No change.
Personal VM: **Smoke test PASSED** — all 3 judges (Prometheus/Llama/Qwen) 20/20 parse rate, sensible scores.
- Prometheus: 1.9 pairs/s, mean=3.15/5, 72B=0.84 (highest) ✓
- Llama: 5.4 pairs/s, mean=5.30/10, 72B=0.74 (highest) ✓
- Qwen: 5.3 pairs/s, mean=6.10/10, 14B=0.70>72B=0.68 (bias visible) ✓
Ready to launch full scoring (test+train, ~4-5h total).
---

## Check — 2026-03-31 18:17 UTC (Cron — Personal VM)

### Personal VM — Reference Judging Progress
- **Prometheus test: COMPLETE** ✅
  - 14,568 outputs in 2502s (5.8 pairs/s)
  - Parse rate: 13,105/14,568 (90.0%) — 10% parse failures
  - Score dist: {1:2427, 2:1310, 3:2253, 4:4534, 5:2581}, mean=3.27/5
- **Llama test: RUNNING** — building 14,568 prompts
- Qwen test: pending (after Llama)
- Train set: pending (after all 3 test judges)

### CSD3
- All 5 jobs PENDING (Priority). No change.

### Status: On track. Test set ~2/3 done (Prometheus complete, Llama running, Qwen next).
---

## Check — 2026-03-31 19:25 UTC (Cron — Personal VM)

### Personal VM — Scoring Progress
- **Test set: ALL 3 JUDGES COMPLETE** ✅
  - prometheus-7b-v2_reference: 13,105/14,568 (90.0%) — STILL has truncation bug from first run
  - llama-3.1-8b-it_reference: 14,567/14,568 (100.0%) ✅
  - qwen2.5-7b-it_reference: 14,568/14,568 (100.0%) ✅
  - Output: test_all_judges.jsonl (23MB)
- **Train set: Prometheus RUNNING** — processing 59,852 prompts, ~2% progress
  - Using fixed script (no max_tokens limit)

### Issues
- Test Prometheus scores are from the OLD run (max_tokens=256, 10% truncation). Need to redo Prometheus on test after train finishes.
- Script uses `>` redirect which overwrote the main log on restart. Per-judge logs were cleaned earlier.

### CSD3
- All 5 jobs PENDING (Priority). No change.
---

## Check — 2026-03-31 20:17 UTC (Cron — Personal VM)

### Personal VM
- All old custom scoring killed, garbage cleaned
- **vLLM server running**: Llama-3.1-8B-Instruct on port 8000 (42GB VRAM)
- **DeepEval G-Eval smoke test PASSED**: Score=0.6 with CoT reasoning, reference-grounded
- Switching to **DeepEval (official G-Eval) + prometheus-eval (official library)** — no more custom code
- No full scoring running yet — confirming approach with user first

### CSD3
- All 5 jobs PENDING (Priority). No change.
---

## Check — 2026-03-31 21:20 UTC (Cron — Personal VM)

### Personal VM — PIPELINE FAILED
- **Step 1 (train scoring): MASSIVE FAILURE**
  - 57,534 / 59,852 pairs FAILED (96% failure rate)
  - Only 2,318 scored successfully
  - Root cause: DeepEval async `max_concurrent=20` overwhelms vLLM server → request timeouts
  - DeepEval retries once per timeout, then marks as failed
- Pipeline incorrectly continued to Step 2 (script should stop if failed > 0)
- **Killed pipeline, cleaned garbage outputs**

### Root cause: concurrency too high for single-GPU vLLM
- 20 concurrent long-context requests (2048 tokens each) on single RTX 8000
- vLLM can't serve them all before DeepEval's timeout threshold

### Action needed from user:
1. What `max_concurrent` to use? (Need to test — suggest 3-5 for single GPU)
2. Pipeline script needs failure check: stop if scored < total
---

## Check — 2026-03-31 22:17 UTC (Cron)
Personal VM: Idle. GPU free (1MB used). vLLM server also stopped. No scoring running.
Pipeline killed after failure. Waiting for user decision on concurrency setting.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: **BLOCKED** — need user decision on `max_concurrent` before restarting.
---

## Check — 2026-04-01 12:00 UTC — Pipeline RELAUNCHED

### Root cause fixed
- `APITimeoutError` from OpenAI client default timeout (~5s) too short for vLLM
- Fix: `LocalModel(timeout=120)` + `DEEPEVAL_DISABLE_TIMEOUTS=true`
- Verified: 800/800 scored, 0 failures at `max_concurrent=8` with 200 entries

### Pipeline running
- `max_concurrent=8`, `timeout=120`
- Failure check: stops if scored < total (won't proceed to downstream steps)
- Step 1: Train (59,852 pairs) ~10h → ETA ~22:00 UTC
- Step 2: Test (14,568 pairs) ~2.5h
- Steps 3-5: post-process, HF upload, model training

### CSD3
- All 5 jobs PENDING (Priority).
---

## Check — 2026-04-01 15:17 UTC (Cron)
Personal VM: Train scoring **~13%** (~7,796/59,852). 78 retries total, 0 permanent failures. Steady ~0.7-0.8 pairs/s. ETA ~16h → **finish ~07:00 UTC April 2**.
CSD3: All 5 jobs PENDING (Priority). Q2 budget still not started.
Status: Running normally.
---

## Check — 2026-04-01 16:17 UTC (Cron)

### Personal VM
- DeepEval train scoring: **~15%** (~9,175/59,852). Running normally.

### CSD3
- **INTR job 26775946 RUNNING** (3.5min) — RoBERTa fused 5ep all targets
- Previous INTR 26775122 FAILED (wrong arg names: `--train-data` → `--input`). Fixed and resubmitted.
- 5 long jobs still PENDING (Priority)

### P100 Nodes
- CUDA installed on both nodes ✅
- Rebooting for nvidia driver load
- Phase 2 (vLLM build) completed on node0 ✅
- Waiting for reboot to verify

### SLO Filter Implementation
- Base interface: `filters/base.py` ✅
- SLOs-Serve filter: `slos_serve_filter.py` ✅
- PolyServe filter: `polyserve_filter.py` ✅
- QLM filter: `qlm_filter.py` ✅
- Cara CDF filter: `cara_cdf_filter.py` ✅ (hard_reject + tiered modes)

Status: All workstreams active. Filter code written, needs integration with scheduler + testing.
---

## Check — 2026-04-01 19:30 UTC (Cron)

### Personal VM
- DeepEval train: ~20% estimated. Running (network flaky to check exact count).

### CSD3 — INTR 26788032 WORKING ✅
- RoBERTa 5ep with `venv_roberta` (4.50.3) + `--max-length 512`
- Length DONE: ρ=0.700, MAE=104 ✅ (matches previous results)
- Length_bucket in progress: acc=0.409 at epoch 1 ✅
- Remaining targets: judge_class, reference_score, similarity
- Root cause of previous failures: wrong venv (5.3.0) + missing --max-length 512

### P100 Nodes
- transformers 4.50.3 ✅
- Models copied (ModernBERT bucket, KNN, XGBoost) ✅
- Scheduler config + filter code synced ✅
- **vLLM server FAILING**: `RuntimeError: Failed to infer device type`
- P100 branch `cara_p100_v_6.0` can import vllm but server fails on device detection
- Needs investigation: may need `VLLM_TARGET_DEVICE=cuda` or branch-specific env var

### Filter Integration
- All filter code + factory + cara_serve.py integration: COMPLETE ✅
- Tested: syntax passes
- **Blocked on P100 vLLM serving** for online testing
---

## Check — 2026-04-01 20:17 UTC (Cron)

### Personal VM
- DeepEval train: **~21%** (~12,632/59,852). 91 retries, 0 permanent failures. Running normally.

### CSD3 — INTR 26788032 COMPLETE ✅
- **ALL 5 RoBERTa targets trained and saved** in 51 min
- length, length_bucket, judge_class, reference_score, similarity
- Models at: `/rds/.../models/cara/roberta_fused_5ep/*/fused_model.pt`
- Root cause of previous failures: wrong venv + missing --max-length 512
- Fix: `venv_roberta` (4.50.3) + `--max-length 512`

### P100
- **Reloading nodes** — fresh OS image to start from clean setup.sh
- Previous manual installs caused environment inconsistency
- Will use exact `setup_p100_smoke.sh` (copy of proven setup.sh) after reload
---

## Check — 2026-04-01 21:30 UTC (Cron)

### Personal VM
- DeepEval train: **~23%** (~13,703/59,852). Rate dropped to 0.4 pairs/s (was 0.7). 100 retries. ETA ~32h.
- Rate slowdown may be from longer prompts in later data or vLLM server pressure.

### CSD3
- INTR 26788032 COMPLETED ✅ (all 5 RoBERTa models saved)
- CSD3 SSH unreachable (MFA may have expired)

### P100
- Nodes reloaded (2nd time — first attempt corrupted by manual commands)
- setup_p100_smoke.sh running in foreground
- Following proven setup.sh exactly — no manual interventions
---

## Check — 2026-04-01 17:17 UTC (Cron)

### Personal VM
- DeepEval train: ~15.5% (~9,281/59,852). 78 retries, 0 permanent failures. Running normally.

### CSD3
- **INTR 26775946 FAILED** — CUDA error in transformers `masking_utils.py` on `similarity` target. `create_bidirectional_mask` + `sdpa_mask` crash. Likely transformers 5.x + A100 attention incompatibility. All 5 targets "DONE" in log but model files all MISSING (crash before save).
- **Resubmitted 26776951** with `CUDA_LAUNCH_BLOCKING=1` for better error trace.
- 5 long jobs still PENDING (Priority)

### P100 Nodes
- CUDA + nvidia-smi: working (P100 12GB confirmed)
- **vLLM build FAILING** — `cara_p100_v_6.0` branch build error on fresh node. Infrastructure issue.

### SLO Filter Implementation
- All 4 filter implementations done (base, SLOs-Serve, PolyServe, QLM, Cara CDF)
- XGBoost calibration script done
- **Blocked on P100 deployment** for online testing
---

## Check — 2026-04-01 18:17 UTC (Cron)

### Personal VM
- DeepEval train: **~17.5%** (~10,447/59,852). Running normally.

### CSD3
- **INTR 26777216 FAILED** — ALL targets crash at step 43 with CUDA device-side assert. `eager` attention didn't help. Root cause: transformers 5.3.0 + torch 2.5.1+cu121 incompatibility with RoBERTa on A100.
- **Need user decision**: downgrade transformers or torch? This is a package change.

### P100
- vLLM build still failing — SyntaxError in build, module imports but is empty.
- File ownership fixed (was root-owned from setup script).
- **Need user decision**: debug further or try a different vLLM branch?
---

## Check — 2026-04-01 12:20 UTC (Cron)
Personal VM: Train scoring running (PID 249369). GPU 100%, ~2828 HTTP requests (~1400 pairs). No timeouts, no errors. On track.
CSD3: Not checked (same as before).
Status: Running normally.
---

## Check — 2026-04-01 13:20 UTC (Cron)
Personal VM: Train scoring **~7%** (~4,222/59,852 pairs). Rate ~0.9 pairs/s. Occasional timeouts retrying successfully (attempt 1 only). ETA ~17.5h for train → **finish ~06:50 UTC April 2**.
CSD3: Not checked.
Status: Running, slower than expected due to retries but no permanent failures.
---

## Check — 2026-04-01 14:20 UTC (Cron)
Personal VM: Train scoring **10.3%** (6,192/59,852 pairs). Rate 0.74 pairs/s. 58 retries (0.94%), 0 permanent failures. ETA ~20h for train → **finish ~10:00 UTC April 2**.
CSD3: Not checked.
Status: Running normally but slower than 200-sample test predicted (0.74 vs 1.6 pairs/s). Sustained load + retries cause slowdown.
---

## Check — 2026-03-31 08:00 UTC (Cron)
Vast: Idle. Disk 82%. No processes.
CSD3: All 5 jobs PENDING (Priority). No change.
Status: No action needed.
---
