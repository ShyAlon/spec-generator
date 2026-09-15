"""Offline replay of byte-exact model completions captured from real runs."""
from __future__ import annotations
import json
from pathlib import Path
from langchain_core.messages import AIMessage

_cursors: dict[tuple[str,str],int]={}

class ReplayChatModel:
    def __init__(self, stage: str, directory: str):
        self.stage=stage;self.directory=str(Path(directory))
    def _next(self):
        files=sorted(Path(self.directory).glob(f"{self.stage}-*.raw.txt"))
        key=(self.directory,self.stage);i=_cursors.get(key,0)
        if i>=len(files): raise RuntimeError(f"replay exhausted for stage {self.stage}: need response {i+1}, have {len(files)}")
        _cursors[key]=i+1
        return AIMessage(content=files[i].read_text(),usage_metadata={"input_tokens":0,"output_tokens":0,"total_tokens":0})
    def invoke(self,messages): return self._next()
    def stream(self,messages):
        msg=self._next();text=msg.content;step=max(1,len(text)//3)
        for i in range(0,len(text),step): yield AIMessage(content=text[i:i+step],usage_metadata=msg.usage_metadata)
