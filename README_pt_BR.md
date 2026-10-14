# Simulação de eletrodiálise 2D

Este projeto simula o transporte de Na⁺ e Cl⁻ em uma célula de eletrodiálise com três canais e duas membranas (CEM e AEM). O modelo utiliza uma discretização pelo método dos volumes finitos, as equações de Nernst–Planck, a eletroneutralidade e condições de interface de Donnan.

## Início rápido

1. Instalar o Python e as dependências:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Alterar os parâmetros em `config.json`.
3. Executar a simulação a partir da pasta do projeto:

   ```bash
   python main.py
   ```

Os resultados são salvos em uma nova subpasta de `resultats/`, identificada pelo nome da simulação e pela data de execução.

## Principais parâmetros (`config.json`)

| Parâmetro | Significado | Exemplo |
| --- | --- | --- |
| `simulation_name` | Nome da simulação | `test_1p5V` |
| `voltage` | Tensão-alvo (V) | `1.5` |
| `ions.Na+.diffusion` | Coeficiente de difusão de Na⁺ (m²/s) | `2.5e-9` |
| `ions.Cl-.diffusion` | Coeficiente de difusão de Cl⁻ (m²/s) | `2.0e-9` |
| `concentration_inlet` | Concentração na entrada (mol/m³) | `500` |
| `temperature` | Temperatura (K) | `298.15` |
| `velocity` | Velocidade média (m/s) | `0.005` |
| `membrane.porosity` | Porosidade das membranas | `0.15` |
| `membrane.charge_cem` | Carga fixa da CEM (conforme a convenção do modelo) | `-1000` |
| `membrane.charge_aem` | Carga fixa da AEM (conforme a convenção do modelo) | `1000` |
| `mesh.nx_regions` | Número de células nas cinco regiões ao longo de x | `[9,6,15,6,9]` |
| `mesh.ny` | Número de células ao longo de y | `18` |
| `newton.tolerance` | Tolerância do solucionador | `4e-7` |
| `newton.max_iterations` | Limite de iterações | `60` |
| `initial_solution` | Caminho opcional para uma solução `.npz` compatível, ou `null` | `null` |
| `generate_plots` | Gerar gráficos | `true` |
| `run_validation` | Executar verificações numéricas | `true` |

Ordem das cinco regiões ao longo de x: canal 1, CEM, canal 2, AEM, canal 3.

Exemplos de malha:

- Rápida: `nx_regions = [3,2,5,2,3]`, `ny = 6` → 15 × 6.
- Intermediária: `nx_regions = [9,6,15,6,9]`, `ny = 18` → 45 × 18.
- Mais refinada: `nx_regions = [15,10,25,10,15]`, `ny = 30` → 75 × 30.

Essas classificações são relativas: ainda é necessário realizar um estudo de convergência de malha para obter resultados quantitativos confiáveis.

## Organização

```text
electrodialysis/
├── main.py                 # Ponto de entrada
├── config.json             # Parâmetros definidos pelo usuário
├── solver.py               # Solucionador físico e numérico
├── continuation.py         # Script de continuação da tensão
├── visualization.py        # Gráficos e diagnósticos
├── validation.py           # Verificações numéricas
├── requirements.txt        # Dependências Python
├── resultats/              # Resultados das simulações
├── references/             # Referências, incluindo COMSOL
├── archives/               # Scripts e gráficos antigos
├── AVANCEMENT_ET_LIMITES.md
└── README.md
```

## Resultados e validação

Para cada simulação convergida, o programa pode salvar `solution.npz`, gráficos em PNG e um diagnóstico na pasta da simulação. As verificações incluem, entre outros aspectos, os balanços de conservação, a eletroneutralidade e as condições de Donnan.

**Importante:** um resíduo pequeno e o cumprimento dos balanços discretos não comprovam nem a convergência de malha nem a validade experimental do modelo.

## Limitações atuais

Atualmente, o solucionador foi desenvolvido para duas espécies monovalentes, Na⁺ e Cl⁻. Simplesmente alterar os nomes dos íons em `config.json` não é suficiente para modelar outros íons, especialmente espécies multivalentes ou radioativas. Seria necessário ampliar o modelo físico e o código.

## Recomendações para compartilhamento

Manter os arquivos de código-fonte e `requirements.txt` no repositório compartilhado. Evitar incluir pastas grandes de resultados, caches do Python e arquivos temporários, a menos que esses resultados sejam necessários para reproduzir um estudo.
