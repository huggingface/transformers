# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for PaliGemma.
"""


import logging
from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import (
    AddedToken,
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from ...utils import TensorType


logger = logging.getLogger(__name__)

IMAGE_TOKEN = "<image>"
EXTRA_TOKENS = ['<loc0000>', '<loc0001>', '<loc0002>', '<loc0003>', '<loc0004>', '<loc0005>', '<loc0006>', '<loc0007>', '<loc0008>', '<loc0009>', '<loc0010>', '<loc0011>', '<loc0012>', '<loc0013>', '<loc0014>', '<loc0015>', '<loc0016>', '<loc0017>', '<loc0018>', '<loc0019>', '<loc0020>', '<loc0021>', '<loc0022>', '<loc0023>', '<loc0024>', '<loc0025>', '<loc0026>', '<loc0027>', '<loc0028>', '<loc0029>', '<loc0030>', '<loc0031>', '<loc0032>', '<loc0033>', '<loc0034>', '<loc0035>', '<loc0036>', '<loc0037>', '<loc0038>', '<loc0039>', '<loc0040>', '<loc0041>', '<loc0042>', '<loc0043>', '<loc0044>', '<loc0045>', '<loc0046>', '<loc0047>', '<loc0048>', '<loc0049>', '<loc0050>', '<loc0051>', '<loc0052>', '<loc0053>', '<loc0054>', '<loc0055>', '<loc0056>', '<loc0057>', '<loc0058>', '<loc0059>', '<loc0060>', '<loc0061>', '<loc0062>', '<loc0063>', '<loc0064>', '<loc0065>', '<loc0066>', '<loc0067>', '<loc0068>', '<loc0069>', '<loc0070>', '<loc0071>', '<loc0072>', '<loc0073>', '<loc0074>', '<loc0075>', '<loc0076>', '<loc0077>', '<loc0078>', '<loc0079>', '<loc0080>', '<loc0081>', '<loc0082>', '<loc0083>', '<loc0084>', '<loc0085>', '<loc0086>', '<loc0087>', '<loc0088>', '<loc0089>', '<loc0090>', '<loc0091>', '<loc0092>', '<loc0093>', '<loc0094>', '<loc0095>', '<loc0096>', '<loc0097>', '<loc0098>', '<loc0099>', '<loc0100>', '<loc0101>', '<loc0102>', '<loc0103>', '<loc0104>', '<loc0105>', '<loc0106>', '<loc0107>', '<loc0108>', '<loc0109>', '<loc0110>', '<loc0111>', '<loc0112>', '<loc0113>', '<loc0114>', '<loc0115>', '<loc0116>', '<loc0117>', '<loc0118>', '<loc0119>', '<loc0120>', '<loc0121>', '<loc0122>', '<loc0123>', '<loc0124>', '<loc0125>', '<loc0126>', '<loc0127>', '<loc0128>', '<loc0129>', '<loc0130>', '<loc0131>', '<loc0132>', '<loc0133>', '<loc0134>', '<loc0135>', '<loc0136>', '<loc0137>', '<loc0138>', '<loc0139>', '<loc0140>', '<loc0141>', '<loc0142>', '<loc0143>', '<loc0144>', '<loc0145>', '<loc0146>', '<loc0147>', '<loc0148>', '<loc0149>', '<loc0150>', '<loc0151>', '<loc0152>', '<loc0153>', '<loc0154>', '<loc0155>', '<loc0156>', '<loc0157>', '<loc0158>', '<loc0159>', '<loc0160>', '<loc0161>', '<loc0162>', '<loc0163>', '<loc0164>', '<loc0165>', '<loc0166>', '<loc0167>', '<loc0168>', '<loc0169>', '<loc0170>', '<loc0171>', '<loc0172>', '<loc0173>', '<loc0174>', '<loc0175>', '<loc0176>', '<loc0177>', '<loc0178>', '<loc0179>', '<loc0180>', '<loc0181>', '<loc0182>', '<loc0183>', '<loc0184>', '<loc0185>', '<loc0186>', '<loc0187>', '<loc0188>', '<loc0189>', '<loc0190>', '<loc0191>', '<loc0192>', '<loc0193>', '<loc0194>', '<loc0195>', '<loc0196>', '<loc0197>', '<loc0198>', '<loc0199>', '<loc0200>', '<loc0201>', '<loc0202>', '<loc0203>', '<loc0204>', '<loc0205>', '<loc0206>', '<loc0207>', '<loc0208>', '<loc0209>', '<loc0210>', '<loc0211>', '<loc0212>', '<loc0213>', '<loc0214>', '<loc0215>', '<loc0216>', '<loc0217>', '<loc0218>', '<loc0219>', '<loc0220>', '<loc0221>', '<loc0222>', '<loc0223>', '<loc0224>', '<loc0225>', '<loc0226>', '<loc0227>', '<loc0228>', '<loc0229>', '<loc0230>', '<loc0231>', '<loc0232>', '<loc0233>', '<loc0234>', '<loc0235>', '<loc0236>', '<loc0237>', '<loc0238>', '<loc0239>', '<loc0240>', '<loc0241>', '<loc0242>', '<loc0243>', '<loc0244>', '<loc0245>', '<loc0246>', '<loc0247>', '<loc0248>', '<loc0249>', '<loc0250>', '<loc0251>', '<loc0252>', '<loc0253>', '<loc0254>', '<loc0255>', '<loc0256>', '<loc0257>', '<loc0258>', '<loc0259>', '<loc0260>', '<loc0261>', '<loc0262>', '<loc0263>', '<loc0264>', '<loc0265>', '<loc0266>', '<loc0267>', '<loc0268>', '<loc0269>', '<loc0270>', '<loc0271>', '<loc0272>', '<loc0273>', '<loc0274>', '<loc0275>', '<loc0276>', '<loc0277>', '<loc0278>', '<loc0279>', '<loc0280>', '<loc0281>', '<loc0282>', '<loc0283>', '<loc0284>', '<loc0285>', '<loc0286>', '<loc0287>', '<loc0288>', '<loc0289>', '<loc0290>', '<loc0291>', '<loc0292>', '<loc0293>', '<loc0294>', '<loc0295>', '<loc0296>', '<loc0297>', '<loc0298>', '<loc0299>', '<loc0300>', '<loc0301>', '<loc0302>', '<loc0303>', '<loc0304>', '<loc0305>', '<loc0306>', '<loc0307>', '<loc0308>', '<loc0309>', '<loc0310>', '<loc0311>', '<loc0312>', '<loc0313>', '<loc0314>', '<loc0315>', '<loc0316>', '<loc0317>', '<loc0318>', '<loc0319>', '<loc0320>', '<loc0321>', '<loc0322>', '<loc0323>', '<loc0324>', '<loc0325>', '<loc0326>', '<loc0327>', '<loc0328>', '<loc0329>', '<loc0330>', '<loc0331>', '<loc0332>', '<loc0333>', '<loc0334>', '<loc0335>', '<loc0336>', '<loc0337>', '<loc0338>', '<loc0339>', '<loc0340>', '<loc0341>', '<loc0342>', '<loc0343>', '<loc0344>', '<loc0345>', '<loc0346>', '<loc0347>', '<loc0348>', '<loc0349>', '<loc0350>', '<loc0351>', '<loc0352>', '<loc0353>', '<loc0354>', '<loc0355>', '<loc0356>', '<loc0357>', '<loc0358>', '<loc0359>', '<loc0360>', '<loc0361>', '<loc0362>', '<loc0363>', '<loc0364>', '<loc0365>', '<loc0366>', '<loc0367>', '<loc0368>', '<loc0369>', '<loc0370>', '<loc0371>', '<loc0372>', '<loc0373>', '<loc0374>', '<loc0375>', '<loc0376>', '<loc0377>', '<loc0378>', '<loc0379>', '<loc0380>', '<loc0381>', '<loc0382>', '<loc0383>', '<loc0384>', '<loc0385>', '<loc0386>', '<loc0387>', '<loc0388>', '<loc0389>', '<loc0390>', '<loc0391>', '<loc0392>', '<loc0393>', '<loc0394>', '<loc0395>', '<loc0396>', '<loc0397>', '<loc0398>', '<loc0399>', '<loc0400>', '<loc0401>', '<loc0402>', '<loc0403>', '<loc0404>', '<loc0405>', '<loc0406>', '<loc0407>', '<loc0408>', '<loc0409>', '<loc0410>', '<loc0411>', '<loc0412>', '<loc0413>', '<loc0414>', '<loc0415>', '<loc0416>', '<loc0417>', '<loc0418>', '<loc0419>', '<loc0420>', '<loc0421>', '<loc0422>', '<loc0423>', '<loc0424>', '<loc0425>', '<loc0426>', '<loc0427>', '<loc0428>', '<loc0429>', '<loc0430>', '<loc0431>', '<loc0432>', '<loc0433>', '<loc0434>', '<loc0435>', '<loc0436>', '<loc0437>', '<loc0438>', '<loc0439>', '<loc0440>', '<loc0441>', '<loc0442>', '<loc0443>', '<loc0444>', '<loc0445>', '<loc0446>', '<loc0447>', '<loc0448>', '<loc0449>', '<loc0450>', '<loc0451>', '<loc0452>', '<loc0453>', '<loc0454>', '<loc0455>', '<loc0456>', '<loc0457>', '<loc0458>', '<loc0459>', '<loc0460>', '<loc0461>', '<loc0462>', '<loc0463>', '<loc0464>', '<loc0465>', '<loc0466>', '<loc0467>', '<loc0468>', '<loc0469>', '<loc0470>', '<loc0471>', '<loc0472>', '<loc0473>', '<loc0474>', '<loc0475>', '<loc0476>', '<loc0477>', '<loc0478>', '<loc0479>', '<loc0480>', '<loc0481>', '<loc0482>', '<loc0483>', '<loc0484>', '<loc0485>', '<loc0486>', '<loc0487>', '<loc0488>', '<loc0489>', '<loc0490>', '<loc0491>', '<loc0492>', '<loc0493>', '<loc0494>', '<loc0495>', '<loc0496>', '<loc0497>', '<loc0498>', '<loc0499>', '<loc0500>', '<loc0501>', '<loc0502>', '<loc0503>', '<loc0504>', '<loc0505>', '<loc0506>', '<loc0507>', '<loc0508>', '<loc0509>', '<loc0510>', '<loc0511>', '<loc0512>', '<loc0513>', '<loc0514>', '<loc0515>', '<loc0516>', '<loc0517>', '<loc0518>', '<loc0519>', '<loc0520>', '<loc0521>', '<loc0522>', '<loc0523>', '<loc0524>', '<loc0525>', '<loc0526>', '<loc0527>', '<loc0528>', '<loc0529>', '<loc0530>', '<loc0531>', '<loc0532>', '<loc0533>', '<loc0534>', '<loc0535>', '<loc0536>', '<loc0537>', '<loc0538>', '<loc0539>', '<loc0540>', '<loc0541>', '<loc0542>', '<loc0543>', '<loc0544>', '<loc0545>', '<loc0546>', '<loc0547>', '<loc0548>', '<loc0549>', '<loc0550>', '<loc0551>', '<loc0552>', '<loc0553>', '<loc0554>', '<loc0555>', '<loc0556>', '<loc0557>', '<loc0558>', '<loc0559>', '<loc0560>', '<loc0561>', '<loc0562>', '<loc0563>', '<loc0564>', '<loc0565>', '<loc0566>', '<loc0567>', '<loc0568>', '<loc0569>', '<loc0570>', '<loc0571>', '<loc0572>', '<loc0573>', '<loc0574>', '<loc0575>', '<loc0576>', '<loc0577>', '<loc0578>', '<loc0579>', '<loc0580>', '<loc0581>', '<loc0582>', '<loc0583>', '<loc0584>', '<loc0585>', '<loc0586>', '<loc0587>', '<loc0588>', '<loc0589>', '<loc0590>', '<loc0591>', '<loc0592>', '<loc0593>', '<loc0594>', '<loc0595>', '<loc0596>', '<loc0597>', '<loc0598>', '<loc0599>', '<loc0600>', '<loc0601>', '<loc0602>', '<loc0603>', '<loc0604>', '<loc0605>', '<loc0606>', '<loc0607>', '<loc0608>', '<loc0609>', '<loc0610>', '<loc0611>', '<loc0612>', '<loc0613>', '<loc0614>', '<loc0615>', '<loc0616>', '<loc0617>', '<loc0618>', '<loc0619>', '<loc0620>', '<loc0621>', '<loc0622>', '<loc0623>', '<loc0624>', '<loc0625>', '<loc0626>', '<loc0627>', '<loc0628>', '<loc0629>', '<loc0630>', '<loc0631>', '<loc0632>', '<loc0633>', '<loc0634>', '<loc0635>', '<loc0636>', '<loc0637>', '<loc0638>', '<loc0639>', '<loc0640>', '<loc0641>', '<loc0642>', '<loc0643>', '<loc0644>', '<loc0645>', '<loc0646>', '<loc0647>', '<loc0648>', '<loc0649>', '<loc0650>', '<loc0651>', '<loc0652>', '<loc0653>', '<loc0654>', '<loc0655>', '<loc0656>', '<loc0657>', '<loc0658>', '<loc0659>', '<loc0660>', '<loc0661>', '<loc0662>', '<loc0663>', '<loc0664>', '<loc0665>', '<loc0666>', '<loc0667>', '<loc0668>', '<loc0669>', '<loc0670>', '<loc0671>', '<loc0672>', '<loc0673>', '<loc0674>', '<loc0675>', '<loc0676>', '<loc0677>', '<loc0678>', '<loc0679>', '<loc0680>', '<loc0681>', '<loc0682>', '<loc0683>', '<loc0684>', '<loc0685>', '<loc0686>', '<loc0687>', '<loc0688>', '<loc0689>', '<loc0690>', '<loc0691>', '<loc0692>', '<loc0693>', '<loc0694>', '<loc0695>', '<loc0696>', '<loc0697>', '<loc0698>', '<loc0699>', '<loc0700>', '<loc0701>', '<loc0702>', '<loc0703>', '<loc0704>', '<loc0705>', '<loc0706>', '<loc0707>', '<loc0708>', '<loc0709>', '<loc0710>', '<loc0711>', '<loc0712>', '<loc0713>', '<loc0714>', '<loc0715>', '<loc0716>', '<loc0717>', '<loc0718>', '<loc0719>', '<loc0720>', '<loc0721>', '<loc0722>', '<loc0723>', '<loc0724>', '<loc0725>', '<loc0726>', '<loc0727>', '<loc0728>', '<loc0729>', '<loc0730>', '<loc0731>', '<loc0732>', '<loc0733>', '<loc0734>', '<loc0735>', '<loc0736>', '<loc0737>', '<loc0738>', '<loc0739>', '<loc0740>', '<loc0741>', '<loc0742>', '<loc0743>', '<loc0744>', '<loc0745>', '<loc0746>', '<loc0747>', '<loc0748>', '<loc0749>', '<loc0750>', '<loc0751>', '<loc0752>', '<loc0753>', '<loc0754>', '<loc0755>', '<loc0756>', '<loc0757>', '<loc0758>', '<loc0759>', '<loc0760>', '<loc0761>', '<loc0762>', '<loc0763>', '<loc0764>', '<loc0765>', '<loc0766>', '<loc0767>', '<loc0768>', '<loc0769>', '<loc0770>', '<loc0771>', '<loc0772>', '<loc0773>', '<loc0774>', '<loc0775>', '<loc0776>', '<loc0777>', '<loc0778>', '<loc0779>', '<loc0780>', '<loc0781>', '<loc0782>', '<loc0783>', '<loc0784>', '<loc0785>', '<loc0786>', '<loc0787>', '<loc0788>', '<loc0789>', '<loc0790>', '<loc0791>', '<loc0792>', '<loc0793>', '<loc0794>', '<loc0795>', '<loc0796>', '<loc0797>', '<loc0798>', '<loc0799>', '<loc0800>', '<loc0801>', '<loc0802>', '<loc0803>', '<loc0804>', '<loc0805>', '<loc0806>', '<loc0807>', '<loc0808>', '<loc0809>', '<loc0810>', '<loc0811>', '<loc0812>', '<loc0813>', '<loc0814>', '<loc0815>', '<loc0816>', '<loc0817>', '<loc0818>', '<loc0819>', '<loc0820>', '<loc0821>', '<loc0822>', '<loc0823>', '<loc0824>', '<loc0825>', '<loc0826>', '<loc0827>', '<loc0828>', '<loc0829>', '<loc0830>', '<loc0831>', '<loc0832>', '<loc0833>', '<loc0834>', '<loc0835>', '<loc0836>', '<loc0837>', '<loc0838>', '<loc0839>', '<loc0840>', '<loc0841>', '<loc0842>', '<loc0843>', '<loc0844>', '<loc0845>', '<loc0846>', '<loc0847>', '<loc0848>', '<loc0849>', '<loc0850>', '<loc0851>', '<loc0852>', '<loc0853>', '<loc0854>', '<loc0855>', '<loc0856>', '<loc0857>', '<loc0858>', '<loc0859>', '<loc0860>', '<loc0861>', '<loc0862>', '<loc0863>', '<loc0864>', '<loc0865>', '<loc0866>', '<loc0867>', '<loc0868>', '<loc0869>', '<loc0870>', '<loc0871>', '<loc0872>', '<loc0873>', '<loc0874>', '<loc0875>', '<loc0876>', '<loc0877>', '<loc0878>', '<loc0879>', '<loc0880>', '<loc0881>', '<loc0882>', '<loc0883>', '<loc0884>', '<loc0885>', '<loc0886>', '<loc0887>', '<loc0888>', '<loc0889>', '<loc0890>', '<loc0891>', '<loc0892>', '<loc0893>', '<loc0894>', '<loc0895>', '<loc0896>', '<loc0897>', '<loc0898>', '<loc0899>', '<loc0900>', '<loc0901>', '<loc0902>', '<loc0903>', '<loc0904>', '<loc0905>', '<loc0906>', '<loc0907>', '<loc0908>', '<loc0909>', '<loc0910>', '<loc0911>', '<loc0912>', '<loc0913>', '<loc0914>', '<loc0915>', '<loc0916>', '<loc0917>', '<loc0918>', '<loc0919>', '<loc0920>', '<loc0921>', '<loc0922>', '<loc0923>', '<loc0924>', '<loc0925>', '<loc0926>', '<loc0927>', '<loc0928>', '<loc0929>', '<loc0930>', '<loc0931>', '<loc0932>', '<loc0933>', '<loc0934>', '<loc0935>', '<loc0936>', '<loc0937>', '<loc0938>', '<loc0939>', '<loc0940>', '<loc0941>', '<loc0942>', '<loc0943>', '<loc0944>', '<loc0945>', '<loc0946>', '<loc0947>', '<loc0948>', '<loc0949>', '<loc0950>', '<loc0951>', '<loc0952>', '<loc0953>', '<loc0954>', '<loc0955>', '<loc0956>', '<loc0957>', '<loc0958>', '<loc0959>', '<loc0960>', '<loc0961>', '<loc0962>', '<loc0963>', '<loc0964>', '<loc0965>', '<loc0966>', '<loc0967>', '<loc0968>', '<loc0969>', '<loc0970>', '<loc0971>', '<loc0972>', '<loc0973>', '<loc0974>', '<loc0975>', '<loc0976>', '<loc0977>', '<loc0978>', '<loc0979>', '<loc0980>', '<loc0981>', '<loc0982>', '<loc0983>', '<loc0984>', '<loc0985>', '<loc0986>', '<loc0987>', '<loc0988>', '<loc0989>', '<loc0990>', '<loc0991>', '<loc0992>', '<loc0993>', '<loc0994>', '<loc0995>', '<loc0996>', '<loc0997>', '<loc0998>', '<loc0999>', '<loc1000>', '<loc1001>', '<loc1002>', '<loc1003>', '<loc1004>', '<loc1005>', '<loc1006>', '<loc1007>', '<loc1008>', '<loc1009>', '<loc1010>', '<loc1011>', '<loc1012>', '<loc1013>', '<loc1014>', '<loc1015>', '<loc1016>', '<loc1017>', '<loc1018>', '<loc1019>', '<loc1020>', '<loc1021>', '<loc1022>', '<loc1023>', '<seg000>', '<seg001>', '<seg002>', '<seg003>', '<seg004>', '<seg005>', '<seg006>', '<seg007>', '<seg008>', '<seg009>', '<seg010>', '<seg011>', '<seg012>', '<seg013>', '<seg014>', '<seg015>', '<seg016>', '<seg017>', '<seg018>', '<seg019>', '<seg020>', '<seg021>', '<seg022>', '<seg023>', '<seg024>', '<seg025>', '<seg026>', '<seg027>', '<seg028>', '<seg029>', '<seg030>', '<seg031>', '<seg032>', '<seg033>', '<seg034>', '<seg035>', '<seg036>', '<seg037>', '<seg038>', '<seg039>', '<seg040>', '<seg041>', '<seg042>', '<seg043>', '<seg044>', '<seg045>', '<seg046>', '<seg047>', '<seg048>', '<seg049>', '<seg050>', '<seg051>', '<seg052>', '<seg053>', '<seg054>', '<seg055>', '<seg056>', '<seg057>', '<seg058>', '<seg059>', '<seg060>', '<seg061>', '<seg062>', '<seg063>', '<seg064>', '<seg065>', '<seg066>', '<seg067>', '<seg068>', '<seg069>', '<seg070>', '<seg071>', '<seg072>', '<seg073>', '<seg074>', '<seg075>', '<seg076>', '<seg077>', '<seg078>', '<seg079>', '<seg080>', '<seg081>', '<seg082>', '<seg083>', '<seg084>', '<seg085>', '<seg086>', '<seg087>', '<seg088>', '<seg089>', '<seg090>', '<seg091>', '<seg092>', '<seg093>', '<seg094>', '<seg095>', '<seg096>', '<seg097>', '<seg098>', '<seg099>', '<seg100>', '<seg101>', '<seg102>', '<seg103>', '<seg104>', '<seg105>', '<seg106>', '<seg107>', '<seg108>', '<seg109>', '<seg110>', '<seg111>', '<seg112>', '<seg113>', '<seg114>', '<seg115>', '<seg116>', '<seg117>', '<seg118>', '<seg119>', '<seg120>', '<seg121>', '<seg122>', '<seg123>', '<seg124>', '<seg125>', '<seg126>', '<seg127>']  # fmt: skip


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def build_string_from_input(prompt, bos_token, image_seq_len, image_token):
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
    )
    The output will be:
    "<im><im><im><s>Initial str"
    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
    """
    return f"{image_token * image_seq_len}{bos_token}{prompt}\n"


class PaliGemmaProcessor(ProcessorMixin):
    r"""
    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

    [`PaliGemmaProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~PaliGemmaProcessor.__call__`] and [`~PaliGemmaProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
        tokens_to_add = {"additional_special_tokens": [image_token]}
        tokenizer.add_special_tokens(tokens_to_add)
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        tokenize_newline_separately: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        do_resize: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional["ChannelDimension"] = "channels_first",  # noqa: F821
        input_data_format: Optional[
            Union[str, "ChannelDimension"]  # noqa: F821
        ] = None,
        resample: "PILImageResampling" = None,  # noqa: F821
        do_convert_rgb: bool = None,
        do_thumbnail: bool = None,
        do_align_long_axis: bool = None,
        do_rescale: bool = None,
        suffix: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        SiglipImageProcessor's [`~SiglipImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        The usage for PaliGemma fine-tuning preparation is slightly different than usual. suffix passed are suffixes to
        the prompt in `text`, and will be placed after the prompt. This is because attention is handled differently for
        the prefix and the suffix. For instance,
        ```python
        image = PIL_cow_image
        prompt = "answer en Where is the cow standing?"
        suffix = "on the beach"
        inputs = processor(text=prompt, images=image, suffix=suffix)
        ```
        Here `inputs` will contain the `input_ids` and `token_type_ids` that follow
        ```python
        inputs["input_ids"][:, 256:]
        # tensor([[     2,   6006,    603,    573,  13910,   9980, 235336,    108,    477,   573,   8318]])
        inputs["token_type_ids"][:, 256:]
        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
        ```
        Meaning the last three tokens are of "label" ("suffix") type while the other ones are of "prefix" type.


        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            tokenize_newline_separately (`bool`, defaults to `True`):
                Adds a separately tokenized '\n' at the end of the prompt.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            suffix (`str`, `List[str]`, `List[List[str]]`):
                The suffixes or batch of suffixes to be encoded. Only necessary for finetuning. See https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md
                for more information. If your prompt is "<image> What is on the image", the suffix corresponds to the expected prediction "a cow sitting on a bench".

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`. If `suffix`
              is provided, the `input_ids` will also contain the suffix input ids.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **labels** -- Labels compatible with training if `suffix` is not None
        """

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PaliGemmaProcessor` instance.")
        if text is None:
            logger.warning_once(
                "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model."
            )
            text = ""

        if isinstance(text, List) and isinstance(images, List):
            if len(images) < len(text):
                raise ValueError(
                    f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image."
                )
        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass
        if suffix is not None and _is_str_or_image(suffix):
            suffix = [suffix]
        if suffix is not None:
            suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]

        input_strings = [
            build_string_from_input(
                prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=IMAGE_TOKEN,
            )
            for prompt in text
        ]

        pixel_values = self.image_processor(
            images,
            do_resize=do_resize,
            do_normalize=do_normalize,
            return_tensors=return_tensors,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
            data_format=data_format,
            resample=resample,
            do_convert_rgb=do_convert_rgb,
        )["pixel_values"]

        if max_length is not None:
            max_length += self.image_seq_length  # max_length has to account for the image tokens

        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_tensors=return_tensors,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_token_type_ids=return_token_type_ids,
        )

        return_data = {**inputs, "pixel_values": pixel_values}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
