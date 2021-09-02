Análise da propensão a mudar de emprego de candidatos a uma posição na area de data science usando técnicas de machine learning

* O caso 

O caso foi o retirado da página: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

Os desafios do caso:
* os dados são desbalanceados
* há muitos dados que não foram preenchidos

A classificação foi feita usando duas técnicas: MLPs e Florestas de árvores de decisão. Os métodos foram escolhidos devido a sua precisão e confiabilidade. MLP é um método bastante robusto e ajustável, sendo ideal para classificações de dados cuja relação entre os parâmetros é complexa. Random Forests também são classificadores bastante robustos e com, geralmente, boa confiabilidade. O uso de árvores de decisão permite também a análise da contribuição dos parâmetros, gerando entendimento de quais fatores são os mais críticos para a classificação.

* Documentação

O projeto foi feito em Python, versão 3.7.0

Foram usadas as bibliotecas:
* sklearn
* pandas
* imblearn

O projeto foi feito no pycharm, então, clonar este repositório e abrir como um novo projeto na referida IDE deve ser o bastante para sua reprodução

* Próximos passos

* Por mais que esteja comentado, reconheço que o código está um pouco bagunçado e uma refatoração poderia ajudar bastante na legibilidade do código 
* Notei posteriormente que muitas funções foram implementadas do zero ao invés de usar métodos disponíveis nas bibliotecas importadas. Uso dessas funções pode deixar a resolução do problema em mais alto nível.
