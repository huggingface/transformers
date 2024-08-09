# تقدير العمق أحادي العين

تقدير العمق أحادي العين هي مهمة رؤية حاسوبية تتضمن التنبؤ بمعلومات العمق لمشهد من صورة واحدة. وبعبارة أخرى، فهي عملية تقدير مسافة الأجسام في مشهد من وجهة نظر كاميرا واحدة.

تقدير العمق أحادي العين له تطبيقات مختلفة، بما في ذلك إعادة الإعمار ثلاثي الأبعاد، والواقع المعزز، والقيادة الذاتية، والروبوتات. إنها مهمة صعبة لأنها تتطلب من النموذج فهم العلاقات المعقدة بين الأجسام في المشهد ومعلومات العمق المقابلة، والتي يمكن أن تتأثر بعوامل مثل ظروف الإضاءة، والاحتجاب، والقوام.

هناك فئتان رئيسيتان لتقدير العمق:

- **تقدير العمق المطلق**: تهدف هذه المهمة المتغيرة إلى توفير قياسات عمق دقيقة من الكاميرا. ويستخدم المصطلح بالتبادل مع تقدير العمق المتري، حيث يتم توفير العمق في قياسات دقيقة بالمتر أو القدم. تخرج نماذج تقدير العمق المطلق خرائط عمق بقيم رقمية تمثل المسافات في العالم الحقيقي.

- **تقدير العمق النسبي**: يهدف تقدير العمق النسبي إلى التنبؤ بترتيب العمق للأجسام أو النقاط في مشهد دون توفير قياسات دقيقة. تخرج هذه النماذج خريطة عمق تشير إلى الأجزاء الأقرب أو الأبعد نسبيًا من المشهد دون المسافات الفعلية إلى A و B.

في هذا الدليل، سنرى كيفية الاستنتاج باستخدام [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)، وهو نموذج تقدير عمق نسبي بدون تصوير فوتوغرافي، و [ZoeDepth](https://huggingface.co/docs/transformers/main/en/model_doc/zoedepth)، وهو نموذج تقدير عمق مطلق.

<Tip>

تحقق من [صفحة مهمة تقدير العمق](https://huggingface.co/tasks/depth-estimation) لعرض جميع التصميمات ونقاط التفتيش المتوافقة.

</Tip>

قبل أن نبدأ، نحتاج إلى تثبيت أحدث إصدار من المحولات:

```bash
pip install -q -U transformers
```

## خط أنابيب تقدير العمق

أبسط طريقة لتجربة الاستنتاج باستخدام نموذج يدعم تقدير العمق هي استخدام خط الأنابيب المقابل [`pipeline`].
قم بتنفيذ خط أنابيب من نقطة تفتيش على [Hub Hugging Face](https://huggingface.co/models؟pipeline_tag=depth-estimation&sort=downloads):

```py
>>> from transformers import pipeline
>>> import torch

>>> device = "cuda" if torch.cuda.is_available() else "cpu"
>>> checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
>>> pipe = pipeline("depth-estimation", model=checkpoint, device=device)
```

بعد ذلك، اختر صورة للتحليل:

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
>>> image = Image.open(requests.get(url, stream=True).raw)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg" alt="Photo of a bee"/>
</div>

مرر الصورة إلى خط الأنابيب.

```py
>>> predictions = pipe(image)
```

مرر الصورة إلى خط الأنابيب.

```py
>>> predictions = pipe(image)
```

يعيد خط الأنابيب قاموسًا يحتوي على إدخالين. الأول، يسمى `predicted_depth`، هو tensor بقيم العمق المعبر عنها بالمتر لكل بكسل.
والثاني، "العمق"، هو صورة PIL التي تصور نتيجة تقدير العمق.

دعونا نلقي نظرة على النتيجة المرئية:

```py
>>> predictions["depth"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization.png" alt="Depth estimation visualization"/>
</div>

## استدلال تقدير العمق باليد

الآن بعد أن رأيت كيفية استخدام خط أنابيب تقدير العمق، دعنا نرى كيف يمكننا تكرار نفس النتيجة باليد.

ابدأ بتحميل النموذج والمعالج المرتبط من نقطة تفتيش على [Hub Hugging Face](https://huggingface.co/models؟pipeline_tag=depth-estimation&sort=downloads).
سنستخدم هنا نفس نقطة التفتيش كما هو الحال من قبل:

```py
>>> from transformers import AutoImageProcessor, AutoModelForDepthEstimation

>>> checkpoint = "Intel/zoedepth-nyu-kitti"

>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
>>> model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)
```

قم بإعداد إدخال الصورة للنموذج باستخدام `image_processor` الذي سيتولى رعاية تحويلات الصور
مثل تغيير الحجم والتوحيد القياسي:

```py
>>> pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
```

مرر المدخلات المحضرة عبر النموذج:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(pixel_values)
```

دعونا نقوم بمعالجة النتائج وتصورها. 

نحن بحاجة إلى إضافة وسادة ثم تغيير حجم الإخراج بحيث يكون لخريطة العمق المتوقعة نفس البعد مثل الصورة الأصلية. بعد تغيير الحجم، سنقوم بإزالة المناطق المبطنة من العمق. 

```py
>>> import numpy as np
>>> import torch.nn.functional as F

>>> predicted_depth = outputs.predicted_depth.unsqueeze(dim=1)
>>> height, width = pixel_values.shape[2:]

>>> height_padding_factor = width_padding_factor = 3
>>> pad_h = int(np.sqrt(height/2) * height_padding_factor)
>>> pad_w = int(np.sqrt(width/2) * width_padding_factor)

>>> if predicted_depth.shape[-2:] != pixel_values.shape[-2:]:
>>>    predicted_depth = F.interpolate(predicted_depth, size= (height, width), mode='bicubic', align_corners=False)

>>> if pad_h > 0:
     predicted_depth = predicted_depth[:, :, pad_h:-pad_h,:]
>>> if pad_w > 0:
     predicted_depth = predicted_depth[:, :, :, pad_w:-pad_w]
```

الآن يمكننا تصور النتائج (تم أخذ الدالة أدناه من إطار عمل [GaussianObject](https://github.com/GaussianObject/GaussianObject/blob/ad6629efadb57902d5f8bc0fa562258029a4bdf1/pred_monodepth.py#L11)).

```py
import matplotlib

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.
Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_partum = (128, 128, 128, 255): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.colormaps.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

>>> result = colorize(predicted_depth.cpu().squeeze().numpy())
>>> Image.fromarray(result)
```



<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/depth-visualization-zoe.png" alt="Depth estimation visualization"/>
</div>