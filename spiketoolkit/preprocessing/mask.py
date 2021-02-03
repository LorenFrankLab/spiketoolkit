from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor
import numpy as np

class MaskRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'Mask'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, mask=None):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        assert mask is not None, ValueError("'mask' should be specified")
        self._mask = np.asarray(mask, dtype='bool')
        BasePreprocessorRecordingExtractor.__init__(self, recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'mask': mask}

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame)
        # check to see if the start or end frame is None, in which case set to 0 and the number of frames, respectively
        if start_frame is None:
            start_frame = 0 
        if end_frame is None:
            end_frame = len(traces[:,0])
        if len(traces[0,:]) == len(mask):
            traces[:,~self._mask[start_frame:end_frame]] = 0.0
        else:  
            ValueError(f'Error: the length of traces {len(traces[:,0])} is different than the length of the mask {len(self._mask)}; no mask applied')
        return traces


def mask(recording, mask=None):
    '''
    Apply a boolean mask to the recording, where False elements of the mask cuase the associated recording frames to
    be set to 0
    
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    mask: numpy array of boolean values of the same length as the number of frames in the recording. False values correspond to frames that will be set to 0.

    Returns
    -------
    masked_traces: MaskTracesRecording
        The masked traces recording extractor object
    '''
    return MaskRecording(
        recording=recording, mask=mask
    )