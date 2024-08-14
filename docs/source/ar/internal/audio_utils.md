# المرافق لبرنامج 'FeatureExtractors'

هذه الصفحة تسرد جميع وظائف المرافق التي يمكن أن يستخدمها برنامج FeatureExtractor الصوتي لحساب ميزات خاصة من الصوت الخام باستخدام خوارزميات شائعة مثل تحويل فورييه قصير الوقت أو مخطط Mel اللوغاريتمي.

معظم هذه الوظائف مفيدة فقط إذا كنت تدرس شفرة معالجات الصوت في المكتبة.

## التحولات الصوتية

[[autodoc]] audio_utils.hertz_to_mel

[[autodoc]] audio_utils.mel_to_hertz

[[autodoc]] audio_utils.mel_filter_bank

[[autodoc]] audio_utils.optimal_fft_length

[[autodoc]] audio_utils.window_function

[[autodoc]] audio_utils.spectrogram

[[autodoc]] audio_utils.power_to_db

[[autodoc]] audio_utils.amplitude_to_db