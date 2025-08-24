# Ai Oluşturma sistemi

size bu sistemde nasıl yapay zeka yapabileceğinizi anlatıcam
projenin yapısı zor gelebilir baktığınızda
ama adım adım anlatıncna anlayacaksınız
Biz bu yapay zeka sisteminde tüm dataları tek bir dosyada birleştirip eğitme yapıcaz
Başka adımları yapmak isteseniz bile yinede tüm adımları okumanızı öneririz
Bilmediğiniz terimler olabilir ama problem etmeyin en sonunda ai yapabileceksiniz

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

Başka zaman projeyi çalıştırmak istediğinizde ilk önce şunları yapmalısınız

## Windows
```sh
.\venv\Scripts\activate
```

## Linux
```sh
.\venv\Scripts\activate
```

# Adım 1, Data bulma
Arkadaşlar öncelikle bize deli dehşet bir data gerekiyor
isterseniz kendi kodlarınızı veya öğretmek istediğiniz dataları jsonl çevirip
ardındnan direk Adım 2'ye geçebilirsiniz.
Biz huggingface de (bigcode/the-stack-v2) die bir proje var arkadaşlar
bu projede örneğin JavaScript yazılım dili için 50gb url dosyası olduğunu düşünün
evet bir txt dosyasına yazdığınızda her karakter 1 byte olduğu için az birşey tutar
ama bu projede 50gb url var. Yani JavaScript dili ile alakalı 50gb code değil 50gb sadece url arkadaşlar. İstediğiniz şekilde data toplayabilirsiniz.

Şimdi size öncelikle 2 data formatı olduğunu söylemem lazım (parquets ve jsonl)
aslında bunlar bildiğiniz bir .json .yml .txt gibi bir data dosyasıdır. okuyabilirsiniz
örneğin jsonl aslında jsonların toplu versionudur, kafanızda büyütmenize gerek yok.
daha basitce adamlar .txt yerine .parquets adında dosya kullanmış düşünebilirsiniz

## Data türünü öğrenme

Şimdi örneğin huggingface den data'lar indirdiniz, öncelikle datanın yapısını bilmeniz gerekir. şimdi şöyle düşünün 
```json
{
    "promt":"nasılsın",
    "content":"iyiyim sen nasılsın?"
}
```
mesela siz üstteki sistemi kullanıyorsunuz varsayalım
ardından başka bir data daha ekliceksiniz varsayalım, alttaki gibi.
```json
{
    "request":"iyimisin",
    "response":"iyiyim, teşekkürler sen nasılsın?"
}
```
fark ettiyseniz promt ve request farklı isimde ama aslında aynı amaç için uğraşıyorlar. arkadaşlar siz bunu böyle eğitmeniz yapay zekanın performansını kötü etkiler. Çünki yapay zeka promt ve request farklı olduğunu düşünür cevaplar saçmalayabilir. indirdiğiniz datanın içini kontol etmeniz gerekir. indirdiğiniz datanın içerisindeki kısımları isimleri değiştirip kullanırsanız problem çıkmayacaktır.

Şimdi kullanıcağımız (bigcode/the-stack-v2) projesinde bir data yok arkadaşlar bunun yerine url ler var bunun içerisinde, sizin indireceğiniz başka projelerde büyük ihtimal data içericektir.

## Url'leri indirme
projeyi indirdiktikten ve ardından vscode açıldıktan sonra .env adında dosya oluşturuyoruz
içerisinde
```.env
HF_TOKEN=BURAYA_TOKENINIZ_GELİCEK
```
bu şekilde yapıştırıyoruz

Şimdi arkadaşlar Huggingface den token (Şifre) almamız gerekicek.
Aldıktan sonra BURAYA_TOKENINIZ_GELİCEK kısmını silip oraya yapıştırın

arama motorunuza (örn google) bigcode/the-stack-v2 yazın ve girin
girdiğiniz huggingface sitesinde "Files and versions" kısmına girin bulamazsanız (ctrl + f) yapın
girdikten sonra aşağıda dosyalar bulucaksınız orada data klasörüne girin ve bu kısımdan istediğiniz bir yazılım dili seçin (modeli eğitirken kullanabileceği url dosyalarını)

örneğin girdiğiniz klasör bu varsayalım
https://huggingface.co/datasets/bigcode/the-stack-v2/tree/main/data/Java
download_js.py dosyasını açalım ve 
eğer başka bir repo seçmek isterseniz repo_id girebilirsiniz

repo_id için url nin "bigcode/the-stack-v2" bu kısmını
subfolder için url nin tree/*/ dan sonrasını "data/Java" kısmını
local_dir ise arkadaşlar modeli bilgisayarınıza indireceği kısım

bunları ayarladıktan sonra `python3 download_js.py` komutunu çalıştıralım indirilsin

## İndirdiğimiz dosyaları jsonl çevirme
Öncelikle şunu söylemem gerekir
Başka bir repo (proje) indirirken tojsonl.py dosyasını editleyip kullanmanız gerekir
Kodu editleyip kullanmadan önce (yapay zekadan yardım alabilirsiniz) 
indirdiğiniz parquet dosyalarının içindeki isimleri öğrenip ona göre jsonl çevirmeniz gerekir. ben sizin ne türde parquet dosyası indirip içinde ne tür datalar olduğunu bilmediğim için tojsonl.py dosyasını (bigcode/the-stack-v2) için hazırladım.

tojsonl.py dosyasını açalım ve ayarlamaları yapalım

### Ayarlar
input_dir = indirdiğimiz url'lerin dosya konumu, örn: `C:\Users\Ai\Desktop\Yeni klasör\data\Java`
output_path = dataları tek dosya formatına (jsonl) çevireceğimiz yeni dosyanın konumu, örn: `C:\Users\Ai\Desktop\Yeni klasör\model.jsonl`
max_threads = Aynı anda indirceği dosya sayısı, örn: `32`
max_retries = Dosyayı indiremez ise tekrar deneme limiti, örn: `5`

#### Raminizin limiti geçmemeli için limitlemeler (değiştirmenize gerek yok)
max_inflight = 256  # aynı anda en fazla bu kadar future bellekte olsun
batch_size = 1000  # daha sık flush, RAM baskısını azaltır

Şuanki yaptığımız kısım indirdiğimiz url'leri alıp gerçek dataları indirmesi ve tek bir dosya formatına getirmesi.
çalışştırmak için `python3 tojsonl.py` dosyasını çalıştıralım

Bu Adım 1 için sistem gerekmez internet ve depolama gerekir başka hiçbirşey gerekmez
cpu,ram,gpu yük binmez sadece bilgiler indirir

Bu aşamanın sonunda isterseniz google drive yedekleyebilirsiniz
```sh
cd drive
pip install google-api-python-client google-auth-oauthlib 
python main.py upload ../model.jsonl
```

# Adım 2, Tokanizer dan geçirme
Burası yapay zekanın hangi kısımlara bakması gerektiğini seçtiğiniz kısım,
Örn: promt olarak nasılsın dedik
jsonl dosyasında nasılsın nerelerde geçtiğinin listesi (53 satır mesela)
Yani sizin gireceğiniz promtlarda hangi datalara bakması gerektiğini kayıt eden yer burası

train_tokanizer.py dosyasındaki yield data["content"] kısmında hangi datalara bakması gerektiğinizi seçmeniz gerekir

## Ayarlar kısmı
Burası direk modelin ne kadar zeki olabileceğini ayarlayan kısım

Örneğin nasılsın içeren dosyalar için
```
min_frequency=örn: datalarda nasılsın 2 kez geçerse sen bunu listeye ekle promtta karşımıza çıkabilir
vocab_size=örn: min_frequency'in eklediği listenin büyüklüğü
```

min_frequency mantığı şu:
İndirdiğiniz datalar örneğin şu:
Selam nasılsın iyimisin
Selam kötüyüm

şimdi burada Selam iki kez geçtiği için listeye örneğin alır ama diğerlerini almaz çünki ondan 2 kez karşılaşmadık
yapay zekaya selam dediğimde artık "Selam kötüyüm" veya "Selam nasılsın iyimisin" diyebilir
ama ben ona nasılsın dediğim zaman cevap veremez cünki o data dan 2 kez geçmedi listeye eklenmedi

vocab_size ne kadar fazla ise model daha zeki olur ama daha fazla ram kullanır
Ayarlamaları yaptıktan sonra `python3 train_model.py` diyip çalıştırabilirsiniz

Örneğin bu aşamada
vocab_size=32000, min_frequency=3, için 64+ GB üstü ram gerekir 25gb'lık data için

Yüksek Cpu ve Yüksek Ram Kullanır. gpu, depolama, internet kullanmaz
Vericeği çıktı tahmini 50mb lık bir dosyadır depolama gerekmez
İşlemci tabanlıdır gpu gerekmez

# Adım 3, Eğitim

