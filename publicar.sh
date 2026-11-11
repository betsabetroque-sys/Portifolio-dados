#!/usr/bin/env bash
# Uso: ./publicar.sh SEU-USUARIO
# Requer: git configurado com SEU e-mail pessoal e o GitHub CLI (gh) logado na conta pessoal.
set -e
USER_GH=$1
[ -z "$USER_GH" ] && echo "Informe seu usuário: ./publicar.sh SEU-USUARIO" && exit 1

for pasta in 01-olist-diagnostico-sql 02-teste-ab-cookie-cats 03-modelo-churn; do
  (cd $pasta && git init -b main && git add . \
    && git -c user.email="$(git config user.email)" commit -m "feat: estrutura inicial do projeto" \
    && gh repo create "$USER_GH/$pasta" --public --source=. --push)
done

# repo do perfil
cp -r seu-usuario perfil-tmp && cd perfil-tmp
git init -b main && git add . && git commit -m "feat: README do perfil"
gh repo create "$USER_GH/$USER_GH" --public --source=. --push
