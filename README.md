# Rede Neural XOR - NumPy

Este projeto implementa uma rede neural simples, do tipo feedforward, utilizando apenas NumPy, com o objetivo de aprender a função lógica XOR.

## Arquitetura da Rede

- **Entrada**: 2 neurônios (valores binários)
- **Camada oculta**: 2 neurônios com ativação `tanh`
- **Saída**: 1 neurônio com ativação `sigmoid`
- **Bias**: incluído por meio da inserção de 1 no vetor de entrada

## Estrutura do Código

- `init_params()`: inicializa os pesos aleatórios
- `forward()`: realiza a propagação dos dados de entrada até a saída
- `backprop()`: calcula os gradientes para ajuste dos pesos
- `update_param()`: aplica os ajustes nos pesos usando gradientes e taxa de aprendizado
- `model()`: executa o treinamento ao longo de múltiplas épocas
- `sig.py`, `tanh.py`, `mse.py`: arquivos auxiliares contendo funções de ativação e erro, com suas derivadas

## Dados de Treinamento

A rede é treinada com o conjunto de dados clássico do problema XOR:

```python
dataset = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]
```

### Resultados Esperados
Após o treinamento, a rede deve ser capaz de se aproximar das saídas corretas do XOR:

``` python
Input: [0 0], Esperado: 0, Predito: ~0.01  
Input: [0 1], Esperado: 1, Predito: ~0.99  
Input: [1 0], Esperado: 1, Predito: ~0.99  
Input: [1 1], Esperado: 0, Predito: ~0.01
```

## Observações:

A rede foi implementada com foco didático, sem o uso de bibliotecas externas de machine learning.

O modelo utiliza vetores de pesos para representar cada neurônio, mantendo a implementação simples e direta.

**O objetivo desse projeto foi tentar entender os fundamentos de redes neurais e backpropagation na prática**
