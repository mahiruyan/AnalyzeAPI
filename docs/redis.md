# Redis Entegrasyonu

## Ortam Değişkeni

`REDIS_URL` değeri tanımlandığında uygulama, analiz sonuçlarını 5 dakikalığına
cache'ler ve isteğe bağlı job kuyruğu (`jobs.queue`) ile çalışabilir. Railway'de
Redis eklentisi açtıktan sonra bağlantı URL'sini ortam değişkeni olarak eklemek
yeterli.

## Cache Davranışı

- Yalnızca yorum toplanmayan (`include_comments=False`) analizler cache'lenir.
- Aynı `platform`, `file_url`, `mode`, başlık/içerik alanları ve
  `engagement_metrics` ile yapılan istekler cache'den döner.
- Redis erişilemezse güvenli şekilde senkron akışa geri düşer.

## Kuyruk Kullanımı

`jobs.queue.enqueue_analysis` yardımıyla istekler Redis listesine eklenebilir.
Redis yapılandırılmamışsa fonksiyon `None` döndürür; böylece mevcut senkron
senaryo bozulmaz. Worker implementasyonu daha sonra genişletilebilir.

