# Akıllı Takip Sistemi — İSG

YOLOv8 tabanlı Kişisel Koruyucu Donanım (KKD) tespit sistemi. Kamera görüntüsü veya video üzerinden çalışanların baret, yelek ve maske takıp takmadığını gerçek zamanlı olarak tespit eder ve ihlalleri loglar.

## Özellikler

- **Gerçek Zamanlı Tespit** — Webcam veya RTSP kamera akışından anlık KKD kontrolü
- **Video Analizi** — Yüklenen video dosyaları üzerinde tespit
- **İhlal Loglama** — Her ihlal `logs/violations.txt` ve `logs/alerts.json` dosyasına kaydedilir
- **Web Arayüzü** — Flask ile tarayıcı üzerinden kullanım
- **Renk Kodlama** — Yeşil: uyumlu, Kırmızı: ihlal, Beyaz: kişi
- **İstatistik Takibi** — Anlık uyumlu/ihlal sayısı `logs/stats.json` dosyasına yazılır

## Tespit Edilen Sınıflar

| Sınıf | Açıklama |
|-------|----------|
| Hardhat / NO-Hardhat | Baret var / yok |
| Safety Vest / NO-Safety Vest | Güvenlik yeleği var / yok |
| Mask / NO-Mask | Maske var / yok |
| Person | Kişi |
| Safety Cone | Güvenlik konisi |
| machinery | Makine |
| vehicle | Araç |

## Kurulum

```bash
git clone https://github.com/ilhamimert/akilli-takip-sistemi-isg.git
cd akilli-takip-sistemi-isg
pip install -r requirements.txt
```

## Kullanım

### Web Arayüzü

```bash
python app.py
```

Tarayıcıda `http://localhost:5000` adresini aç.

- **Ana Sayfa** — Video dosyası yükle ve analiz et
- **Webcam** — Canlı kamera akışını izle

### Komut Satırı (Webcam)

```bash
python detect.py --source 0
```

### Komut Satırı (Video veya RTSP)

```bash
python detect.py --source video.mp4
python detect.py --source rtsp://kamera-ip/stream
```

### Parametreler

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--source` | `0` | Kaynak: 0=webcam, video yolu, RTSP URL |
| `--conf` | `0.5` | Güven eşiği (0.0 - 1.0) |
| `--imgsz` | `640` | Görüntü boyutu |

## Loglar

```
logs/
├── violations.txt   ← Metin formatında ihlal kayıtları
├── alerts.json      ← Son 10 ihlal (web arayüzü için)
└── stats.json       ← Anlık uyumlu/ihlal sayısı
```

## Teknik Stack

| Bileşen | Teknoloji |
|---------|-----------|
| Nesne Tespiti | YOLOv8 (Ultralytics) |
| Görüntü İşleme | OpenCV |
| Web Arayüzü | Flask |
| Model | `models/best.pt` (özel eğitilmiş) |

## Gereksinimler

- Python 3.10+
- Webcam veya IP kamera (canlı tespit için)