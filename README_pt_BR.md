# Simulação de Eletrodiálise 2D

## 1. Para que serve este projeto?

Este projeto permite simular o funcionamento de uma célula de eletrodiálise utilizando Python.

A eletrodiálise é uma técnica que utiliza corrente elétrica e membranas para transportar íons presentes na água.

Na nossa simulação, estudamos dois íons:
- **Na⁺**: íon sódio.
- **Cl⁻**: íon cloreto.

Nossa célula possui **três canais e duas membranas**:

- **CEM**: membrana que permite principalmente a passagem de íons positivos (cátions).
- **AEM**: membrana que permite principalmente a passagem de íons negativos (ânions).

O programa calcula, entre outras grandezas:
- A concentração dos íons na célula.
- O potencial elétrico.
- O transporte dos íons através das membranas.
- A corrente elétrica.

O objetivo é compreender como diferentes parâmetros, como a tensão elétrica ou as propriedades das membranas, influenciam o funcionamento da célula.

## 2. Como instalar o projeto?

### Etapa 1 — Baixar o projeto

Abra o repositório no GitHub e clique em:

**Code → Download ZIP**

Em seguida, extraia o arquivo ZIP no computador.

Também é possível obter o projeto utilizando Git:

```bash
git clone https://github.com/tom29lg-glitch/electrodialysis.git
```

### Etapa 2 — Instalar o Python

É necessário ter o Python instalado no computador.

Para verificar, abra um terminal e digite:

```bash
python --version
```

Se o Python não estiver instalado, faça o download em https://www.python.org/downloads/.

Em alguns computadores, especialmente nos Macs, o comando utilizado é `python3` em vez de `python`.

### Etapa 3 — Instalar as bibliotecas necessárias

Abra a pasta do projeto no VS Code e, em seguida, abra um terminal.

Execute:

```bash
python -m pip install -r requirements.txt
```

Esse comando instala as bibliotecas necessárias para o funcionamento do programa, incluindo NumPy, SciPy e Matplotlib.

Normalmente, essa instalação precisa ser feita apenas uma vez.

## 3. Como executar uma simulação?

Basta seguir três etapas.

**Etapa 1 — Abrir `config.json`**

Esse arquivo contém todos os parâmetros que podem ser modificados.

Por exemplo:

```json
"simulation_name": "test_1p5V",
"voltage": 1.5,
"concentration_inlet": 500,
"temperature": 298.15,
"velocity": 0.005
```

Esses parâmetros correspondem a:
- `simulation_name`: nome dado à simulação.
- `voltage`: tensão elétrica aplicada, em volts.
- `concentration_inlet`: concentração dos íons na entrada.
- `temperature`: temperatura em kelvin.
- `velocity`: velocidade média da água nos canais.

Para uma primeira simulação, recomenda-se manter os parâmetros que já estão definidos no arquivo.

**Etapa 2 — Executar o programa**

No terminal do VS Code, dentro da pasta do projeto:

```bash
python main.py
```

**Etapa 3 — Aguardar o término do cálculo**

O programa realiza automaticamente os cálculos necessários.

Para facilitar a convergência numérica, ele pode começar com uma tensão baixa e aumentá-la gradualmente até atingir a tensão desejada.

Quando a simulação termina, o programa apresenta a seguinte pergunta:

```text
Voulez-vous sauvegarder les résultats ? [O/n] :
```

- Pressione **Enter** ou digite `o` para manter os resultados.
- Digite `n` para excluir os resultados dessa simulação.

*Observação: a mensagem aparece em francês porque está definida dessa forma no código Python.*

## 4. Onde encontrar os resultados?

Os resultados são salvos automaticamente na pasta `resultats/`.

Cada simulação possui sua própria subpasta.

Por exemplo:

```text
resultats/
└── test_1p5V_45x18_2026-10-15_14h30m00s/
    ├── config.json
    ├── solution.npz
    ├── export.csv
    ├── calcul.log
    └── figures/
```

**Para que servem esses arquivos?**

| Arquivo | Função |
|---|---|
| `config.json` | Armazena os parâmetros utilizados na simulação. |
| `solution.npz` | Contém todos os resultados numéricos da simulação. |
| `export.csv` | Permite visualizar e analisar os resultados no Excel ou Python. |
| `calcul.log` | Contém informações sobre a execução e a convergência do cálculo. |
| `figures/` | Contém os gráficos gerados automaticamente. |

Os gráficos permitem visualizar, entre outras informações, as concentrações de Na⁺ e Cl⁻, o potencial elétrico e os perfis ao longo dos canais.

## 5. Quais parâmetros podem ser modificados?

Todos os principais parâmetros estão no arquivo `config.json`.

| Parâmetro | Para que serve? |
|---|---|
| `voltage` | Modificar a tensão aplicada. |
| `ions.Na+.diffusion` | Modificar o coeficiente de difusão do sódio. |
| `ions.Cl-.diffusion` | Modificar o coeficiente de difusão do cloreto. |
| `concentration_inlet` | Modificar a concentração na entrada. |
| `temperature` | Modificar a temperatura. |
| `velocity` | Modificar a velocidade média da água. |
| `membrane.porosity` | Modificar a porosidade das membranas. |
| `membrane.charge_cem` | Modificar a carga fixa da membrana CEM. |
| `membrane.charge_aem` | Modificar a carga fixa da membrana AEM. |
| `mesh` | Modificar a resolução da malha numérica. |
| `generate_plots` | Ativar ou desativar a geração de gráficos. |
| `generate_csv` | Ativar ou desativar a exportação para CSV. |
| `run_validation` | Ativar ou desativar as verificações numéricas. |

Para os parâmetros físicos, é importante respeitar as unidades indicadas no arquivo de configuração.

**Atenção:** o modelo atual foi desenvolvido para os íons Na⁺ e Cl⁻. Ainda não é possível simular corretamente outros íons apenas alterando seus nomes.

## 6. Como escolher a malha numérica?

A malha numérica corresponde à divisão do domínio em pequenas células utilizadas para realizar os cálculos.

Quanto mais refinada for a malha, maior poderá ser a precisão da simulação, mas também maior será o tempo necessário para os cálculos.

No arquivo `config.json`, os parâmetros são:

```json
"nx_regions": [9, 6, 15, 6, 9],
"ny": 18
```

Os cinco números de `nx_regions` correspondem às cinco regiões do modelo:

**Canal 1 → CEM → Canal 2 → AEM → Canal 3**

Veja três exemplos:

| Malha | Parâmetros | Utilização |
|---|---|---|
| 15 × 6 | `[3,2,5,2,3]`, `ny=6` | Testes muito rápidos |
| 45 × 18 | `[9,6,15,6,9]`, `ny=18` | Simulações habituais |
| 75 × 30 | `[15,10,25,10,15]`, `ny=30` | Cálculos mais detalhados |

**Para começar, recomenda-se utilizar a malha 45 × 18.**

Uma malha mais refinada não garante automaticamente um resultado exato: é necessário comparar diferentes malhas para verificar a convergência numérica.

## 7. Para que servem os diferentes scripts?

| Arquivo | Função |
|---|---|
| `main.py` | Inicia e organiza toda a simulação. |
| `config.json` | Contém os parâmetros que podem ser modificados. |
| `solver.py` | Realiza os cálculos físicos e numéricos. |
| `visualization.py` | Gera os gráficos. |
| `validation.py` | Verifica determinados resultados numéricos. |
| `export_csv.py` | Exporta os resultados para um arquivo CSV. |
| `continuation.py` | Script antigo de continuação da tensão, mantido separadamente. |
| `requirements.txt` | Lista as bibliotecas Python necessárias. |

**Para uma utilização normal, basta modificar `config.json` e executar `main.py`.** Não é necessário modificar os outros scripts.

## 8. Como trabalhar em equipe utilizando o GitHub?

Para obter as últimas modificações do projeto:

```bash
git pull
```

Para enviar suas próprias modificações, recomenda-se criar uma branch e propor um Pull Request no GitHub.

Os arquivos de resultados normalmente são mantidos em cada computador e não são enviados automaticamente para o GitHub.

## 9. Quais são as limitações do modelo?

O programa utiliza um método numérico chamado **método dos volumes finitos** para resolver as equações de transporte dos íons.

O modelo considera, entre outros fenômenos, a difusão, a migração elétrica, o transporte associado ao escoamento e as propriedades das membranas.

No entanto:
- Atualmente, o modelo está limitado a dois íons monovalentes: Na⁺ e Cl⁻.
- Ele ainda não está adaptado para simular íons radioativos ou multivalentes.
- Os resultados precisam ser verificados, especialmente por meio da comparação entre diferentes malhas numéricas.
- A convergência numérica, por si só, não garante que o modelo represente perfeitamente um experimento real.
