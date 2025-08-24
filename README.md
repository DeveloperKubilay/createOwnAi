# Ai Oluşturma sistemi

size bu sistemde nasıl yapay zeka yapabileceğinizi anlatıcam<br>
projenin yapısı zor gelebilir baktığınızda<br>
ama adım adım anlatıncna anlayacaksınız<br>
Biz bu yapay zeka sisteminde tüm dataları tek bir dosyada birleştirip eğitme yapıcaz<br>
Başka adımları yapmak isteseniz bile yinede tüm adımları okumanızı öneririz<br>
Bilmediğiniz terimler olabilir ama problem etmeyin en sonunda ai yapabileceksiniz<br>

# Projeyi kendi bilgisayarınıza yükleme

## Projeyi indirme
```sh
git clone https://github.com/DeveloperKubilay/howToMakeAi
cd howToMakeAi
code .
python -m venv venv
```

## Windows
```sh
.\venv\Scripts\activate
```

## Linux
```sh
.\venv\Scripts\activate
```

## Gerekli yazılımları indirme
```sh
pip install -r requirements.txt
```

Başka zaman projeyi çalıştırmak istediğinizde ilk önce şunları yapmalısınız<br>

## Windows
```sh
.\venv\Scripts\activate
```

## Linux
```sh
.\venv\Scripts\activate
```

# Adım 1, Data bulma
Arkadaşlar öncelikle bize deli dehşet bir data gerekiyor<br>
isterseniz kendi kodlarınızı veya öğretmek istediğiniz dataları jsonl çevirip<br>
ardındnan direk Adım 2'ye geçebilirsiniz.<br>
Biz huggingface de (bigcode/the-stack-v2) die bir proje var arkadaşlar<br>
bu projede örneğin JavaScript yazılım dili için 50gb url dosyası olduğunu düşünün<br>
evet bir txt dosyasına yazdığınızda her karakter 1 byte olduğu için az birşey tutar<br>
ama bu projede 50gb url var. Yani JavaScript dili ile alakalı 50gb code değil 50gb sadece url arkadaşlar. İstediğiniz şekilde data toplayabilirsiniz.<br><br>

Şimdi size öncelikle 2 data formatı olduğunu söylemem lazım (parquets ve jsonl)<br>
aslında bunlar bildiğiniz bir .json .yml .txt gibi bir data dosyasıdır. okuyabilirsiniz<br>
örneğin jsonl aslında jsonların toplu versionudur, kafanızda büyütmenize gerek yok.<br>
daha basitce adamlar .txt yerine .parquets adında dosya kullanmış düşünebilirsiniz<br>

## Data türünü öğrenme

Şimdi örneğin huggingface den data'lar indirdiniz, öncelikle datanın yapısını bilmeniz gerekir. şimdi şöyle düşünün <br>
```json
{
    "promt":"nasılsın",
    "content":"iyiyim sen nasılsın?"
}
```
mesela siz üstteki sistemi kullanıyorsunuz varsayalım<br>
ardından başka bir data daha ekliceksiniz varsayalım, alttaki gibi.<br>
```json
{
    "request":"iyimisin",
    "response":"iyiyim, teşekkürler sen nasılsın?"
}
```
fark ettiyseniz promt ve request farklı isimde ama aslında aynı amaç için uğraşıyorlar. arkadaşlar siz bunu böyle<br>
eğitmeniz yapay zekanın performansını kötü etkiler. Çünki yapay zeka promt ve request farklı olduğunu düşünür cevaplar <br>
saçmalayabilir. indirdiğiniz datanın içini kontol etmeniz gerekir. indirdiğiniz datanın içerisindeki kısımları isimleri <br>
değiştirip kullanırsanız problem çıkmayacaktır.<br>

Şimdi kullanıcağımız (bigcode/the-stack-v2) projesinde bir data yok arkadaşlar bunun yerine url ler var bunun içerisinde, sizin<br> indireceğiniz başka projelerde büyük ihtimal data içericektir.<br>

## Url'leri indirme
projeyi indirdiktikten ve ardından vscode açıldıktan sonra .env adında dosya oluşturuyoruz<br>
içerisinde<br>
```.env
HF_TOKEN=BURAYA_TOKENINIZ_GELİCEK
```
bu şekilde yapıştırıyoruz<br>

Şimdi arkadaşlar Huggingface den token (Şifre) almamız gerekicek.<br>
Aldıktan sonra BURAYA_TOKENINIZ_GELİCEK kısmını silip oraya yapıştırın<br>

arama motorunuza (örn google) bigcode/the-stack-v2 yazın ve girin<br>
girdiğiniz huggingface sitesinde "Files and versions" kısmına girin bulamazsanız (ctrl + f) yapın<br>
girdikten sonra aşağıda dosyalar bulucaksınız orada data klasörüne girin ve bu kısımdan istediğiniz bir yazılım dili seçin (modeli eğitirken kullanabileceği url dosyalarını)<br>

örneğin girdiğiniz klasör bu varsayalım<br>
https://huggingface.co/datasets/bigcode/the-stack-v2/tree/main/data/Java<br>
download_js.py dosyasını açalım ve <br>
eğer başka bir repo seçmek isterseniz repo_id girebilirsiniz<br>

repo_id için url nin "bigcode/the-stack-v2" bu kısmını<br>
subfolder için url nin tree/*/ dan sonrasını "data/Java" kısmını<br>
local_dir ise arkadaşlar modeli bilgisayarınıza indireceği kısım<br>

bunları ayarladıktan sonra `python3 download_js.py` komutunu çalıştıralım indirilsin<br>

## İndirdiğimiz dosyaları jsonl çevirme
Öncelikle şunu söylemem gerekir<br>
Başka bir repo (proje) indirirken tojsonl.py dosyasını editleyip kullanmanız gerekir<br>
Kodu editleyip kullanmadan önce (yapay zekadan yardım alabilirsiniz) <br>
indirdiğiniz parquet dosyalarının içindeki isimleri öğrenip ona göre jsonl çevirmeniz gerekir. ben sizin ne türde parquet dosyası indirip içinde ne tür datalar olduğunu bilmediğim için tojsonl.py dosyasını (bigcode/the-stack-v2) için hazırladım.<br>

tojsonl.py dosyasını açalım ve ayarlamaları yapalım<br>

### Ayarlar<br>
input_dir = indirdiğimiz url'lerin dosya konumu, örn: `C:\Users\Ai\Desktop\Yeni klasör\data\Java`<br>
output_path = dataları tek dosya formatına (jsonl) çevireceğimiz yeni dosyanın konumu, örn: `C:\Users\Ai\Desktop\Yeni klasör\model.jsonl`<br>
max_threads = Aynı anda indirceği dosya sayısı, örn: `32`<br>
max_retries = Dosyayı indiremez ise tekrar deneme limiti, örn: `5`<br>

#### Raminizin limiti geçmemeli için limitlemeler (değiştirmenize gerek yok)<br>
max_inflight = 256  # aynı anda en fazla bu kadar future bellekte olsun<br>
batch_size = 1000  # daha sık flush, RAM baskısını azaltır<br>

Şuanki yaptığımız kısım indirdiğimiz url'leri alıp gerçek dataları indirmesi ve tek bir dosya formatına getirmesi.<br>
çalışştırmak için `python3 tojsonl.py` dosyasını çalıştıralım<br>

Bu Adım 1 için sistem gerekmez internet ve depolama gerekir başka hiçbirşey gerekmez<br>
cpu,ram,gpu yük binmez sadece bilgiler indirir<br>

Bu aşamanın sonunda isterseniz google drive yedekleyebilirsiniz<br>
```sh
cd drive
pip install google-api-python-client google-auth-oauthlib 
python main.py upload ../model.jsonl
```

# Adım 2, Tokanizer dan geçirme
Burası yapay zekanın hangi kısımlara bakması gerektiğini seçtiğiniz kısım,<br>
Örn: promt olarak nasılsın dedik<br>
jsonl dosyasında nasılsın nerelerde geçtiğinin listesi (53 satır mesela)<br>
Yani sizin gireceğiniz promtlarda hangi datalara bakması gerektiğini kayıt eden yer burası<br>

train_tokanizer.py dosyasındaki yield data["content"] kısmında hangi datalara bakması gerektiğinizi seçmeniz gerekir<br>

## Ayarlar kısmı<br>
Burası direk modelin ne kadar zeki olabileceğini ayarlayan kısım<br>

Örneğin nasılsın içeren dosyalar için<br>
```
min_frequency=örn: datalarda nasılsın 2 kez geçerse sen bunu listeye ekle promtta karşımıza çıkabilir
vocab_size=örn: min_frequency'in eklediği listenin büyüklüğü
```

min_frequency mantığı şu:<br>
İndirdiğiniz datalar örneğin şu:<br>
Selam nasılsın iyimisin<br>
Selam kötüyüm<br>

şimdi burada Selam iki kez geçtiği için listeye örneğin alır ama diğerlerini almaz çünki ondan 2 kez karşılaşmadık<br>
yapay zekaya selam dediğimde artık "Selam kötüyüm" veya "Selam nasılsın iyimisin" diyebilir<br>
ama ben ona nasılsın dediğim zaman cevap veremez cünki o data dan 2 kez geçmedi listeye eklenmedi<br>

vocab_size ne kadar fazla ise model daha zeki olur ama daha fazla ram kullanır<br>
Ayarlamaları yaptıktan sonra `python3 train_model.py` diyip çalıştırabilirsiniz<br>

Örneğin bu aşamada<br>
vocab_size=32000, min_frequency=3, için 64+ GB üstü ram gerekir 25gb'lık data için<br>

Yüksek Cpu ve Yüksek Ram Kullanır. gpu, depolama, internet kullanmaz<br>
Vericeği çıktı tahmini 50mb lık bir dosyadır depolama gerekmez<br>
İşlemci tabanlıdır gpu gerekmez<br>

# Adım 3, Eğitim

