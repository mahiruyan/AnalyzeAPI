# Metrik Normalizasyon Kuralları

Bu doküman Instagram ve TikTok metriklerini tek bir şemaya map'leme kurallarını açıklar.

## Platform Metrikleri

### Instagram
- `plays` → `plays` (video izlenme sayısı)
- `reach` → `reach` (erişim sayısı)
- `likes` → `likes` (beğeni sayısı)
- `comments` → `comments` (yorum sayısı)
- `shares` → `shares` (paylaşım sayısı)
- `saves` → `saves` (kaydetme sayısı)
- `avg_watch_time_s` → `avg_watch_time_s` (ortalama izleme süresi)
- `completion_rate` → `completion_rate` (tamamlama oranı)

### TikTok
- `views` → `views` (görüntülenme sayısı)
- `likes` → `likes` (beğeni sayısı)
- `comments` → `comments` (yorum sayısı)
- `shares` → `shares` (paylaşım sayısı)
- `avg_watch_time_s` → `avg_watch_time_s` (ortalama izleme süresi)
- `completion_rate` → `completion_rate` (tamamlama oranı)

## Normalizasyon Kuralları

### Base Metric (Ana Metrik)
- Instagram: `plays` kullanılır
- TikTok: `views` kullanılır
- Skorlama: `min(metric / 10000, 1.0) * 30` (max 30 puan)

### Engagement Score
- Instagram: `likes + comments + shares + saves`
- TikTok: `likes + comments + shares`
- Skorlama: `(engagement_total / max(base_metric, 1)) * 40` (max 40 puan)

### Watch Time Bonus
- `avg_watch_time_s`: `min(watch_time / 30, 1.0) * 15` (max 15 puan)
- `completion_rate`: `completion_rate * 15` (max 15 puan)

## Veri Gecikmesi

### Erken Pencere: 5-30 dakika
- İlk metrikler genellikle 5-30 dakika içinde gelir
- Bu süre zarfında skorlar düşük olabilir
- Tam metrikler 24-48 saat sonra stabil hale gelir

### Öneriler
- Erken skorları dikkate alırken dikkatli olun
- 24 saat sonraki skorlar daha güvenilirdir
- Trend analizi için en az 3-7 günlük veri gerekir

## Skor Aralıkları

- **0-39**: Low (Düşük performans)
- **40-69**: Mid (Orta performans)
- **70-100**: High (Yüksek performans)

## API Endpoint'leri

### POST /ingest_ig
Instagram metriklerini alır ve normalize eder.

### POST /ingest_tt
TikTok metriklerini alır ve normalize eder.

### POST /analyze
Normalize edilmiş payload ile skor hesaplar ve öneriler sunar.
