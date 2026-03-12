# voice-segmentation

Biblioteca modular de segmentação de voz com múltiplas estratégias intercambiáveis.
Cada segmentador expõe a mesma interface e se integra a um pipeline que cuida de IO,
pós-processamento de duração e metadados.

## Estratégias disponíveis

| Pipeline | Método | Extra necessário |
|---|---|---|
| `WebRTCPipeline` | WebRTC VAD clássico | `[webrtc]` |
| `SilencePipeline` | Detecção de silêncios via RMS (librosa) | — |
| `SileroPipeline` | Silero VAD (rede neural, leve) | `[silero]` |
| `FireRedPipeline` | FireRed VAD (rede neural, pesos baixados automaticamente) | `[fireredvad]` |

## Instalação

Instalação base — inclui `SilencePipeline` e dependências de IO:

```bash
pip install voice-segmentation
```

Com os extras desejados:

```bash
pip install "voice-segmentation[webrtc]"            # WebRTC VAD
pip install "voice-segmentation[silero]"            # Silero VAD (torch)
pip install "voice-segmentation[fireredvad]"        # FireRed VAD (torch + pesos HuggingFace)
pip install "voice-segmentation[all]"               # tudo
```

## Uso rápido

```python
from voice_segmentation import SileroPipeline

pipeline = SileroPipeline(threshold=0.5, soft_lower=2.0, soft_upper=20.0)
result = pipeline.run("audio.flac", output="segments/")

for seg in result.segments:
    print(f"{seg.start:.2f}s – {seg.end:.2f}s  →  {seg.path}")
```

Todas as pipelines aceitam os mesmos parâmetros de duração (`soft_lower`, `soft_upper`,
`hard_lower`, `hard_upper`, `overlap`, `max_gap`) e as mesmas opções de saída em `run()`.

## Adicionando uma nova estratégia

Implemente o protocolo `Segmenter` e subclasse `Pipeline` — são as únicas exigências:

```python
from voice_segmentation.pipelines.base import Pipeline, Segmenter
from voice_segmentation.settings import DurationSettings
from voice_segmentation.types import AudioArray, Segment


class MySegmenter:
    def segment(self, audio: AudioArray, sample_rate: int, settings: DurationSettings) -> list[Segment]:
        ...  # sua lógica aqui


class MyPipeline(Pipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._segmenter = MySegmenter()

    @property
    def _segmenter_settings(self):
        return None
```

## Desenvolvimento

```bash
make install-dev   # instala dependências + hooks
make test          # roda a suíte de testes
make lint          # ruff check
make typecheck     # mypy
```

## Contribuindo

Contribuições são bem-vindas. O processo é:

1. **Fork** o repositório e crie uma branch descritiva (`feat/meu-segmentador`, `fix/bug-x`).

2. **Configure o ambiente:**
   ```bash
   pip install -e ".[all,dev,test]"
   pre-commit install
   ```

3. **Faça as alterações.** Para um novo segmentador, implemente o protocolo descrito
   em [Adicionando uma nova estratégia](#adicionando-uma-nova-estratégia) acima.
   Se a implementação requer dependências pesadas (torch, modelos), adicione-a como
   um extra opcional em `pyproject.toml`.

4. **Valide antes do PR:**
   ```bash
   make test       # todos os testes devem passar
   make lint       # sem erros de linter
   make typecheck  # sem erros de tipo
   ```

5. **Commits** seguem o padrão [Conventional Commits](https://www.conventionalcommits.org/):
   `feat:`, `fix:`, `test:`, `chore:`, `docs:`.

6. Abra o **Pull Request** com uma descrição clara do que foi alterado e por quê.

## Licença

MIT
