# AI_кабанчики 
Команда "AI_кабанчики", кейс Депстрой: **14 место из 45**
## Варианты развертывания
### Dockerhub
1. Скачать с Dockerhub контейнер по ссылке: 
   `osetr4/hackaton-app:latest`
2. Запустить контейнер
### Ручной
1. Скачать веса модели  
   `https://drive.google.com/drive/folders/17Pk_Rm7bhy1fC1PkYv418zEAvTgIq5By?usp=sharing`
2. Разместить веса в папке app
3. Перейти в директорию проекта
4. Запустить команду для сборки `docker build -t app ./app`
5. Запустить команду для запуска контейнера `docker run {Название image}`
6. Перейти в интерфейс на _localhost_
