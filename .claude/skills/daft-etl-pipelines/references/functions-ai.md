# functions ai

## classify_image

```python
classify_image(image: Expression, labels: Label | list[Label], *, provider: str | Provider | None=None, model: str | None=None, **options: Unpack[ClassifyImageOptions]) -> Expression
```

Returns an expression that classifies images using the specified model and provider.

Args:
    image (Image Expression):
        The input image column expression.
    labels (str | list[str]):
        Label(s) for classification.
    provider (str | Provider | None):
        The provider to use for the embedding model.
        By default this will use 'transformers' provider
    model (str | None):
        The classifier model to use. Can be a model instance or a model name.
        By default this will use `zero-shot-classification` model
    **options:
        Any additional options to pass for the model.

Note:
    Make sure the required provider packages are installed (e.g. vllm, transformers, openai).

Returns:
    Expression (String Expression): An expression representing the most-probable label string.

## classify_text

```python
classify_text(text: Expression, labels: Label | list[Label], *, provider: str | Provider | None=None, model: str | None=None, **options: Unpack[ClassifyTextOptions]) -> Expression
```

Returns an expression that classifies text using the specified model and provider.

Args:
    text (String Expression):
        The input text column expression.
    labels (str | list[str]):
        Label(s) for classification.
    provider (str | Provider | None):
        The provider to use for the embedding model.
        By default this will use 'transformers' provider
    model (str | None):
        The classifier model to use. Can be a model instance or a model name.
        By default this will use `zero-shot-classification` model
    **options:
        Any additional options to pass for the model.

Note:
    Make sure the required provider packages are installed (e.g. vllm, transformers, openai).

Returns:
    Expression (String Expression): An expression representing the most-probable label string.

## embed_image

```python
embed_image(image: Expression, *, provider: str | Provider | None=None, model: str | None=None, **options: Unpack[EmbedImageOptions]) -> Expression
```

Returns an expression that embeds images using the specified image model and provider.

Args:
    image (Image Expression): The input image column expression.
    provider (str | Provider | None): The provider to use for the image model. If None, the default provider is used.
    model (str | None): The image model to use. Can be a model instance or a model name. If None, the default model is used.
    **options: Any additional options to pass for the model.

Note:
    Make sure the required provider packages are installed (e.g. vllm, transformers, openai).

Returns:
    Expression (Embedding Expression): An expression representing the embedded image vectors.

## embed_text

```python
embed_text(text: Expression, *, provider: str | Provider | None=None, model: str | None=None, dimensions: int | None=None, **options: Unpack[EmbedTextOptions]) -> Expression
```

Returns an expression that embeds text using the specified embedding model and provider.

Args:
    text (String Expression):
        The input text column expression.
    provider (str | Provider | None):
        The provider to use for the embedding model. If None, the default provider is used.
    model (str | None):
        The embedding model to use. Can be a model instance or a model name. If None, the default model is used.
    dimensions (int | None):
        Number of dimensions the output embeddings should have, if the provider and model support specifying. If None, will use the default for the model.
    **options: Any additional options to pass for the model.

Note:
    Make sure the required provider packages are installed (e.g. vllm, transformers, openai).

Returns:
    Expression (Embedding Expression): An expression representing the embedded text vectors.

## jev

```python
jev(row: Expression, condition: str, threshold: float=0.5, *, model: str | None=None, api_key: str | None=None, base_url: str | None=None, timeout: float=60.0) -> Expression
```

Whether each row satisfies a plain-English condition, i.e. `jev_prob(row, condition) >= threshold`.

Args:
    row (Expression): The value to judge. Pass `to_struct(...)` to judge several columns; field names are visible to the model.
    condition (str): A plain-English condition, e.g. "the name is European".
    threshold (float): The minimum probability that counts as true. Defaults to 0.5.
    model (str | None): The Jev model. Defaults to `TYPESAFE_DEFAULT_MODEL`, else `jev-latest`.
    api_key (str | None): TypeSafe API key. Defaults to `TYPESAFE_API_KEY`.
    base_url (str | None): API root, e.g. to route through a proxy. Defaults to `TYPESAFE_BASE_URL`, else `https://api.typesafe.ai`.
    timeout (float): Seconds allowed for each request, which carries up to 25 rows. Defaults to 60.

Returns:
    Expression: A Boolean, or null where `row` is null.

## jev_choice

```python
jev_choice(row: Expression, question: str, options: list[str], *, model: str | None=None, api_key: str | None=None, base_url: str | None=None, timeout: float=60.0) -> Expression
```

The most probable of `options` as the answer to `question` for each row.

Args:
    row (Expression): The value to classify. Pass `to_struct(...)` to classify on several columns.
    question (str): A plain-English question, e.g. "which team should handle this?".
    options (list[str]): Between 1 and 255 distinct candidate answers; duplicates are dropped.
    model (str | None): The Jev model. Defaults to `TYPESAFE_DEFAULT_MODEL`, else `jev-latest`.
    api_key (str | None): TypeSafe API key. Defaults to `TYPESAFE_API_KEY`.
    base_url (str | None): API root, e.g. to route through a proxy. Defaults to `TYPESAFE_BASE_URL`, else `https://api.typesafe.ai`.
    timeout (float): Seconds allowed for each request, which carries up to 25 rows. Defaults to 60.

Returns:
    Expression: A String drawn from `options`, or null where `row` is null.

## jev_prob

```python
jev_prob(row: Expression, condition: str, *, model: str | None=None, api_key: str | None=None, base_url: str | None=None, timeout: float=60.0) -> Expression
```

Probability that each row satisfies a plain-English condition.

Args:
    row (Expression): The value to judge. Pass `to_struct(...)` to judge several columns; field names are visible to the model.
    condition (str): A plain-English condition, e.g. "the customer is angry".
    model (str | None): The Jev model. Defaults to `TYPESAFE_DEFAULT_MODEL`, else `jev-latest`.
    api_key (str | None): TypeSafe API key. Defaults to `TYPESAFE_API_KEY`.
    base_url (str | None): API root, e.g. to route through a proxy. Defaults to `TYPESAFE_BASE_URL`, else `https://api.typesafe.ai`.
    timeout (float): Seconds allowed for each request, which carries up to 25 rows. Defaults to 60.

Returns:
    Expression: A Float64 probability in [0, 1], or null where `row` is null.

## jev_score

```python
jev_score(row: Expression, question: str, levels: list[str], *, model: str | None=None, api_key: str | None=None, base_url: str | None=None, timeout: float=60.0) -> Expression
```

Where each row falls on an ordered rubric, as a probability-weighted level index.

Args:
    row (Expression): The value to score. Pass `to_struct(...)` to score on several columns.
    question (str): A plain-English question, e.g. "how luxurious is this product?".
    levels (list[str]): Between 2 and 10 rubric levels, lowest first.
    model (str | None): The Jev model. Defaults to `TYPESAFE_DEFAULT_MODEL`, else `jev-latest`.
    api_key (str | None): TypeSafe API key. Defaults to `TYPESAFE_API_KEY`.
    base_url (str | None): API root, e.g. to route through a proxy. Defaults to `TYPESAFE_BASE_URL`, else `https://api.typesafe.ai`.
    timeout (float): Seconds allowed for each request, which carries up to 25 rows. Defaults to 60.

Returns:
    Expression: A Float64 in [0, len(levels) - 1] that may fall between levels, or null where `row` is null.

## prompt

```python
prompt(messages: list[Expression] | Expression, return_format: BaseModel | None=None, *, system_message: str | None=None, provider: str | Provider | None=None, model: str | None=None, **options: Any) -> Expression
```

Returns an expression that prompts a large language model using the specified model and provider.

Args:
    messages (list[Expression] | Expression): The list of messages to prompt the model with. Each expression can be either:
        - Plain text strings (always treated as input_text)
        - Image data (numpy arrays, bytes, or File objects - detected by MIME type)
        - Files (PDF, TXT, HTML, audio, video, etc.) as bytes or File objects (detected by MIME type)
    return_format (BaseModel | None): The return format for the prompt. Use a Pydantic model for structured outputs.
    system_message (str | None): The system message for the prompt.
    provider (str | Provider | None): The provider to use for the prompt (default: "openai").
    model (str | None): The model to use for the prompt.
    **options: Any additional options to pass for the prompt.

Returns:
    Expression (String Expression): An expression representing the prompt result.
