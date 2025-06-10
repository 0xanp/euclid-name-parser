docker stop euclid-name-parser || true
docker rm euclid-name-parser || true
docker rmi euclid-name-parser || true
docker build -t euclid-name-parser .
docker run -d -p 3001:8501 --name euclid-name-parser euclid-name-parser
