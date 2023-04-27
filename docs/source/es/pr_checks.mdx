<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Verificaciones en un Pull Request

Cuando abres un _pull request_ en 🤗 Transformers, se ejecutarán una serie de verificaciones para asegurarte de que el _patch_ que estás agregando no rompa nada existente. Estas verificaciones son de cuatro tipos:
- pruebas regulares
- creación de la documentación
- estilo del código y documentación
- consistencia del repositorio

En este documento, intentaremos explicar cuáles son esas diferentes verificaciones y el motivo detrás de ellas, así como también cómo depurarlas localmente si una falla en tu PR.

Recuerda que todas las verificaciones requieren que tengas una instalación de desarrollo:

```bash
pip install transformers[dev]
```

o una instalación editable:

```bash
pip install -e .[dev]
```

del repositorio de Transformers.

## Pruebas

Todos los procesos que comienzan con `ci/circleci: run_tests_` ejecutan partes del conjunto de pruebas de Transformers. Cada uno de esos procesos se enfoca en una parte de la biblioteca en un entorno determinado: por ejemplo, `ci/circleci: run_tests_pipelines_tf` ejecuta la prueba de _pipelines_ en un entorno donde solo está instalado TensorFlow.

Ten en cuenta que para evitar ejecutar pruebas cuando no hay un cambio real en los módulos que estás probando, solo se ejecuta una parte del conjunto de pruebas: se ejecuta una tarea auxiliar para determinar las diferencias en la biblioteca antes y después del PR (lo que GitHub te muestra en la pestaña "Files changes") y selecciona las pruebas afectadas por esa diferencia. Este auxiliar se puede ejecutar localmente usando:

```bash
python utils/tests_fetcher.py
```

desde el directorio raiz del repositorio de Transformers. Se ejecutará lo siguiente:

1. Verificación para cada archivo en el _diff_ si los cambios están en el código, solo en comentarios o _docstrings_. Solo los archivos con cambios reales de código se conservan.
2. Creación de un mapa interno que proporciona para cada archivo del código fuente de la biblioteca todos los archivos a los que impacta recursivamente. Se dice que el módulo A impacta al módulo B si el módulo B importa el módulo A. Para el impacto recursivo, necesitamos una cadena de módulos que va del módulo A al módulo B en la que cada módulo importa el anterior.
3. Aplicación de este mapa en los archivos recopilados en el paso 1, lo que nos da una lista de archivos modelo afectados por el PR.
4. Asignación de cada uno de esos archivos a sus archivos de prueba correspondientes y para obtener una la lista de pruebas a ejecutar.

Al ejecutar el _script_ localmente, debes obtener los resultados de los pasos 1, 3 y 4 impresos y así saber qué pruebas se ejecutarán. El _script_ también creará un archivo llamado `test_list.txt` que contiene la lista de pruebas para ejecutar, y puede ejecutarlas localmente con el siguiente comando:

```bash
python -m pytest -n 8 --dist=loadfile -rA -s $(cat test_list.txt)
```

En caso de que se te escape algo, el conjunto completo de pruebas también se ejecuta a diario.

## Creación de la documentación

El proceso `build_pr_documentation` compila y genera una vista previa de la documentación para asegurarse de que todo se vea bien una vez que se fusione tu PR. Un bot agregará un enlace para obtener una vista previa de la documentación en tu PR. Cualquier cambio que realices en el PR se actualiza automáticamente en la vista previa. Si la documentación no se genera, haz clic en **Detalles** junto al proceso fallido para ver dónde salió mal. A menudo, el error es tan simple como que falta un archivo en `toctree`.

Si estás interesado en compilar u obtener una vista previa de la documentación localmente, echa un vistazo al [`README.md`](https://github.com/huggingface/transformers/tree/main/docs) en la carpeta `docs`.

## Estilo de código y documentación.

El formato de código se aplica a todos los archivos fuente, los ejemplos y las pruebas utilizando `black` e `ruff`. También tenemos una herramienta personalizada que se ocupa del formato de los _docstrings_ y archivos `rst` (`utils/style_doc.py`), así como del orden de las importaciones _lazy_ realizadas en los archivos `__init__.py` de Transformers (`utils /custom_init_isort.py`). Todo esto se puede probar ejecutando

```bash
make style
```

CI verifica que se hayan aplicado dentro de la verificación `ci/circleci: check_code_quality`. También se ejecuta `ruff`, que hará una verificación básica a tu código y te hará saber si encuentra una variable no definida, o una que no se usa. Para ejecutar esa verificación localmente, usa

```bash
make quality
```

Esto puede llevar mucho tiempo, así que para ejecutar lo mismo solo en los archivos que modificaste en la rama actual, ejecuta

```bash
make fixup
```

Este último comando también ejecutará todas las verificaciones adicionales para la consistencia del repositorio. Echemos un vistazo a estas pruebas.

## Consistencia del repositorio

Esta verificación reagrupa todas las pruebas para asegurarse de que tu PR deja el repositorio en buen estado, y se realiza mediante `ci/circleci: check_repository_consistency`. Puedes ejecutar localmente esta verificación ejecutando lo siguiente:

```bash
make repo-consistency
```

Esta instrucción verifica que:

- Todos los objetos agregados al _init_ están documentados (realizados por `utils/check_repo.py`)
- Todos los archivos `__init__.py` tienen el mismo contenido en sus dos secciones (realizado por `utils/check_inits.py`)
- Todo el código identificado como una copia de otro módulo es consistente con el original (realizado por `utils/check_copies.py`)
- Todas las clases de configuración tienen al menos _checkpoint_ válido mencionado en sus _docstrings_ (realizado por `utils/check_config_docstrings.py`)
- Las traducciones de los README y el índice del documento tienen la misma lista de modelos que el README principal (realizado por `utils/check_copies.py`)
- Las tablas generadas automaticamente en la documentación están actualizadas (realizadas por `utils/check_table.py`)
- La biblioteca tiene todos los objetos disponibles incluso si no están instaladas todas las dependencias opcionales (realizadas por `utils/check_dummies.py`)

Si esta verificación falla, los primeros dos elementos requieren una reparación manual, los últimos cuatro pueden repararse automáticamente ejecutando el comando

```bash
make fix-copies
```

Las verificaciones adicionales se refieren a los PRs que agregan nuevos modelos, principalmente que:

- Todos los modelos agregados están en un Auto-mapping (realizado por `utils/check_repo.py`)
<!-- TODO Sylvain, add a check that makes sure the common tests are implemented.-->
- Todos los modelos se verifican correctamente (realizados por `utils/check_repo.py`)

<!-- TODO Sylvain, add the following
- All models are added to the main README, inside the main doc
- All checkpoints used actually exist on the Hub

-->
