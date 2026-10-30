# التنبؤ بالإصابة بمرض السكري باستخدام الشبكات العصبونية

مشروع ANN601 للفصل F25 مطابق للمقترح المقبول عن بيانات Pima Indians Diabetes.

## التشغيل المحلي

استخدم Python 3.12 وافتح الطرفية داخل مجلد Code:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe verify_results.py
.\.venv\Scripts\python.exe predict.py --test-position 0
```

على Linux أو macOS استخدم `python3.12` ثم `.venv/bin/python` بدلا من المسار الخاص بويندوز. التدريب يعمل على CPU ولا يحتاج GPU أو حساب Kaggle أو اتصال إنترنت بعد تثبيت المكتبات. الأمر train.py يعيد إنشاء محتويات results.

## Google Colab

Colab اختياري وليس شرطا في التكليف. افتح https://colab.research.google.com ثم File ثم Upload notebook واختر ANN601_Diabetes_Colab.ipynb الموجود في مجلد التسليم. اختر Runtime ثم Run all. البيانات والكود مضمنان في الدفتر؛ لا تحتاج رفع ملفات أخرى أو ربط Google Drive. الخلية الأخيرة تنزل ملف النتائج المضغوط. أعد تشغيل بيئة Colab إذا طلبت ذلك بعد تثبيت المكتبات. لم ترفع الملفات إلى حساب Google بواسطة هذا المشروع.

## البيانات والمعالجة

الملف data/diabetes.csv يحتوي 768 سجلا و8 ميزات والخرج Outcome. عدد الفئة 0 هو 500، والفئة 1 هو 268. فهرسة الصفوف تبدأ من الصفر. مصدر الفهرسة المذكور في المقترح:
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

استعملت نسخة CSV المنشورة من نفس بيانات Pima في مستودع Datasets لأن رابط التنزيل من Kaggle أعاد 404 وقت الإعداد. لم تستبدل مجموعة البيانات بمجموعة أخرى. أضيف سطر أسماء الأعمدة إلى النسخة العددية فقط، وبقيت قيم الصفوف وترتيبها كما في المصدر:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

وصف البيانات الأصلي المحفوظ:
https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names

توجد بصمة SHA256 ورابط التنزيل في data/source.json. ملف CSV مرفق حتى يبقى التشغيل مستقلا عن توفر الروابط.

نحول الصفر إلى قيمة مفقودة في Glucose وBloodPressure وSkinThickness وInsulin وBMI فقط، ولا نحذف صفوفا. الصفر في Pregnancies قيمة صحيحة. نقسم طبقيا ببذرة 42 إلى 460 تدريبا و154 تحققا و154 اختبارا. نتعلم وسيط التعويض ومتوسط وانحراف StandardScaler من التدريب فقط. نفحص القيم المتطرفة بحدود IQR المحسوبة على التدريب ونبقيها لعدم وجود دليل على خطئها.

## النموذجان

- MLP_A: ثمانية مداخل، طبقة مخفية من 32، وخرج واحد. عدد المعاملات 321.
- MLP_B: ثمانية مداخل، طبقتان من 32 و16، وخرج واحد. عدد المعاملات 833.

التفعيل المخفي ReLU والخرج Sigmoid، والمحسن Adam ومعدل التعلم 0.001 وحجم الدفعة 32 وalpha=0.0001. نستعمل MLPClassifier مع partial_fit للحفاظ على حالة Adam. خسارة التحقق تختار الدورة والبنية، ثم نقيم النموذجين على الاختبار المحجوز. لم نغير الإعدادات لتحسين نتيجة الاختبار.

الحد الأقصى 500 دورة. يتوقف التدريب بعد 40 دورة دون تحسن مهم بمقدار 0.0001 في خسارة التحقق. نحفظ أفضل دورة عند أي انخفاض في خسارة التحقق، حتى إذا كان أصغر من ذلك المقدار. عتبة التصنيف ثابتة عند 0.5.

## النتائج المحفوظة

اختير MLP_B على التحقق: خسارته 0.416266 مقابل 0.418626 للشبكة MLP_A. دقة الاختبار 71.43% مقابل 69.48%. استدعاء الفئة المصابة 51.85% مقابل 46.30%، وF1 يساوي 0.5600 مقابل 0.5155. التفاصيل في results/metrics.json وcomparison.csv. هذه تجربة تعليمية على مجموعة محدودة ولا تثبت صلاحية تشخيصية.

## الملفات

- train.py: تحميل البيانات والمعالجة والتدريب والمقارنة والرسوم.
- predict.py: مثال تنبؤ باستخدام النموذج المختار وكل خطوات المعالجة المحفوظة.
- verify_results.py: إعادة حساب المقاييس وفحص التقسيم والمعالجة والتنبؤات.
- results: النماذج المدربة والسجلات والتنبؤات والرسوم والقياسات.

مثال إدخال قياسات بنفس ترتيب الميزات في FEATURES:

```powershell
.\.venv\Scripts\python.exe predict.py --values 6 148 72 35 0 33.6 0.627 50
```

هذا المثال صف معروف من مجموعة البيانات، وليس حالة مستقلة جديدة. يحمل predict.py ملفات joblib المحلية المرفقة فقط؛ لا تحمل ملفات نماذج من مصدر مجهول.

المراجع البرمجية: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html وhttps://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html وhttps://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
