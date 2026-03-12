"""Configurações de entrada e saída do pipeline de segmentação."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class IOSettings(BaseModel):
    """Configurações de escrita de segmentos e metadados.

    Attributes:
        output_folder: Diretório raiz onde as saídas serao gravadas.
        create_run_subfolder: Se True, cria um subdiretório por execução dentro de output_folder.
            O nome e derivado do arquivo de origem e do timestamp UTC, ou de run_name quando
            fornecido.
        create_segment_subfolders: Se True, cada segmento e gravado em seu próprio subdiretório
            (segment_000/, segment_001/, ...). O subdiretório contem o arquivo de áudio com nome
            áudio.<formato> e, opcionalmente, um metadata.json individual.
        write_run_metadata: Se True, grava run_metadata.json na raiz do diretório da execução com
            estatísticas globais e a lista completa de segmentos.
        write_segment_metadata: Se True, grava um arquivo metadata.json individual por segmento,
            dentro do subdiretório do segmento ou adjacente ao áudio conforme
            create_segment_subfolders.
        redo_if_exists: Se True, recria o diretório de saída quando ele ja existir, descartando
            resultados anteriores. Se False, a execução é pulada e o caminho existente é retornado
            sem modificação.
        audio_format: Formato de áudio para os segmentos gravados. flac é sem perdas e
            recomendado; wav é compatível com ferramentas legadas.
        run_name: Nome personalizado do subdiretório da execução. Quando None, o nome é gerado
            automaticamente como {stem}_{YYYYMMDD_HHMMSS}.
        segment_prefix: Prefixo dos nomes de arquivo e subdiretório de cada segmento. Quando
            None, usa o stem do arquivo de origem. O nome final segue o padrão
            {prefix}_segment_{NNN}.{formato} ou {prefix}_segment_{NNN}/ para subdiretórios.
    """

    output_folder: Path = Field(description="Diretório raiz de saída")
    create_run_subfolder: bool = Field(
        default=True,
        description="Cria subdiretório por execução dentro de output_folder",
    )
    create_segment_subfolders: bool = Field(
        default=False,
        description="Cada segmento e gravado em seu próprio subdiretório",
    )
    write_run_metadata: bool = Field(
        default=True,
        description="Grava run_metadata.json na raiz da execução",
    )
    write_segment_metadata: bool = Field(
        default=True,
        description="Grava metadata.json por segmento",
    )
    redo_if_exists: bool = Field(
        default=True,
        description="Recria o diretório de saída se ja existir; caso contrario pula",
    )
    audio_format: Literal["flac", "wav"] = Field(
        default="flac",
        description="Formato de áudio dos segmentos gravados (flac ou wav)",
    )
    run_name: str | None = Field(
        default=None,
        description="Nome personalizado do subdiretório da execução (None = automático)",
    )
    segment_prefix: str | None = Field(
        default=None,
        description="Prefixo dos arquivos de segmento (None = stem do arquivo de origem)",
    )
    write_segment_spectrograms: bool = Field(
        default=False,
        description=(
            "Salva uma imagem PNG com três espectrogramas (STFT, Mel, CQT) por segmento "
            "ao lado do arquivo de áudio."
        ),
    )
