from .bandpass_filter import bandpass_filter, BandpassFilterRecording
from .notch_filter import notch_filter, NotchFilterRecording
from .whiten import whiten, WhitenRecording
from .common_reference import common_reference, CommonReferenceRecording
from .resample import resample, ResampleRecording
from .rectify import rectify, RectifyRecording
from .remove_artifacts import remove_artifacts, RemoveArtifactsRecording
from .mask import mask, MaskRecording
from .transform import transform, TransformRecording
from .remove_bad_channels import remove_bad_channels, RemoveBadChannelsRecording
from .normalize_by_quantile import normalize_by_quantile, NormalizeByQuantileRecording
from .clip import clip, ClipRecording
from .blank_saturation import blank_saturation, BlankSaturationRecording
from .center import center, CenterRecording

preprocessers_full_list = [
    BandpassFilterRecording,
    NotchFilterRecording,
    WhitenRecording,
    CommonReferenceRecording,
    ResampleRecording,
    RectifyRecording,
    RemoveArtifactsRecording,
    RemoveBadChannelsRecording,
    TransformRecording,
    NormalizeByQuantileRecording,
    ClipRecording,
    BlankSaturationRecording,
    CenterRecording
]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
preprocesser_dict = {pp_class.preprocessor_name: pp_class for pp_class in preprocessers_full_list}
