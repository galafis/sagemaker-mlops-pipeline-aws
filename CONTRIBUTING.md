# Contribuindo / Contributing

## Português

Obrigado pelo interesse em contribuir! Siga os passos abaixo:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Instale as dependências: `make install`
4. Faça suas alterações seguindo o padrão do projeto
5. Execute os testes: `make test`
6. Verifique o estilo: `make lint`
7. Commit suas alterações (`git commit -m 'Adiciona nova feature'`)
8. Push para a branch (`git push origin feature/nova-feature`)
9. Abra um Pull Request

### Padrões de Código

- **Formatação:** Black (line-length=100)
- **Imports:** isort
- **Linting:** flake8
- **Type hints:** obrigatórios em todas as funções públicas
- **Testes:** pytest com cobertura mínima de 80%

---

## English

Thank you for your interest in contributing! Follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Install dependencies: `make install`
4. Make your changes following the project conventions
5. Run tests: `make test`
6. Check code style: `make lint`
7. Commit your changes (`git commit -m 'Add new feature'`)
8. Push to the branch (`git push origin feature/new-feature`)
9. Open a Pull Request

### Code Standards

- **Formatting:** Black (line-length=100)
- **Imports:** isort
- **Linting:** flake8
- **Type hints:** required on all public functions
- **Tests:** pytest with minimum 80% coverage
