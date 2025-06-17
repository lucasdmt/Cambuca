import subprocess

entrada = "chaves_mascaradas.txt"
saida = "chave.txt"
executavel = "./Cambuca"

with open(entrada, "r") as f:
    linhas = f.readlines()

for i, linha in enumerate(linhas, start=1):
    # Salva a linha atual em chave.txt
    with open(saida, "w") as f_out:
        f_out.write(linha.strip() + "\n")

    print(f"▶️ Rodando linha {i}/{len(linhas)}...")

    # Executa o programa Cambuca
    try:
        resultado = subprocess.run([executavel], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar ./Cambuca na linha {i}")
    except FileNotFoundError:
        print(f"❌ Arquivo {executavel} não encontrado. Verifique se está no diretório correto.")
        break

print("✅ Finalizado.")
