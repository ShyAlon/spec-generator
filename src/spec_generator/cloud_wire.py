"""Cloud child wire setup: every provider call over the reviewed admitted wire.

When the runner hands this process an admission descriptor (VMODEL_ADMISSION_FD),
model traffic must go through SDK clients whose transport reserves the exact
outgoing bytes through that descriptor and settles raw usage back over it.
This module installs that binding for the lifetime of one cli run.

No descriptor means a local run: this module does nothing and local behavior
is unchanged. A descriptor that cannot be set up is fatal: a cloud run never
proceeds unbound, and no call silently falls back to message-shaped admission
(the parent serves exact-byte reservations only). The local Ollama route needs
no provider key in the child: the SDK reaches the loopback server directly and
only admission crosses the descriptor. Other providers stay held here - a
child process has no reviewed key path for them.
"""
import os
import urllib.parse
from contextlib import contextmanager

import httpx


@contextmanager
def admitted_child_run():
    """Bind the descriptor's gateway and the wire clients for one cli run.

    No-op without VMODEL_ADMISSION_FD. With it, any setup failure raises
    before the run starts (fail closed), the binding covers the whole run,
    and the descriptor is closed exactly once at the end.
    """
    fd = os.environ.get('VMODEL_ADMISSION_FD')
    if not fd:
        yield
        return
    from .admission_client import SocketAdmissionGateway
    from .attempt_admission import admission_gateway
    from .wire_binding import wire_clients
    gateway = SocketAdmissionGateway(int(fd))  # an invalid descriptor fails here, never mid-run
    try:
        with admission_gateway(gateway, required=True), wire_clients(gateway, _ollama_wire_factory()):
            yield
    finally:
        gateway.close()


def _ollama_wire_factory():
    """The child-side client factory: job-bound Ollama SDK clients whose
    transports admit exact bytes through the run's gateway.

    Origin and path come from OLLAMA_BASE_URL (the same setting the engines
    read), never from a request. Redirects, ambient proxy config and alternate
    mounts are refused. The factory rejects any non-Ollama provider: the
    descriptor wire carries admission, not credentials, and a child has no
    reviewed key path for a keyed provider.
    """
    base = (os.environ.get('OLLAMA_BASE_URL') or 'http://localhost:11434/v1').strip().rstrip('/')
    parsed = urllib.parse.urlsplit(base)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('plain reviewed Ollama base URL required')
    origin = f'{parsed.scheme}://{parsed.hostname}' + (':' + str(parsed.port) if parsed.port not in (None, 443) else '')
    allowed_path = parsed.path + '/chat/completions'

    def _check(request):
        if str(request.url.copy_with(path='', query=None, fragment=None)) not in (origin, origin + '/'):
            raise ValueError('fixed Ollama SDK origin required')
        if request.method != 'POST' or request.url.path != allowed_path:
            raise ValueError('fixed Ollama SDK path required')

    class _SyncFixed(httpx.BaseTransport):
        def __init__(self):
            self.inner = httpx.HTTPTransport(retries=0)

        def handle_request(self, request):
            _check(request)
            response = self.inner.handle_request(request)
            if 300 <= response.status_code < 400:
                response.close()
                raise ValueError('redirect forbidden')
            return response

        def close(self):
            self.inner.close()

    class _AsyncFixed(httpx.AsyncBaseTransport):
        def __init__(self):
            self.inner = httpx.AsyncHTTPTransport(retries=0)

        async def handle_async_request(self, request):
            _check(request)
            response = await self.inner.handle_async_request(request)
            if 300 <= response.status_code < 400:
                await response.aclose()
                raise ValueError('redirect forbidden')
            return response

        async def aclose(self):
            await self.inner.aclose()

    def factory(*, stage, provider, model, gateway):
        if provider != 'ollama':
            raise ValueError('the cloud child wire installs the local Ollama route only')
        from .provider_transport import AdmittedTransport, AsyncAdmittedTransport
        return (httpx.Client(transport=AdmittedTransport(_SyncFixed(), gateway, stage=stage, provider=provider, model=model),
                             trust_env=False, follow_redirects=False),
                httpx.AsyncClient(transport=AsyncAdmittedTransport(_AsyncFixed(), gateway, stage=stage, provider=provider, model=model),
                                  trust_env=False, follow_redirects=False))

    return factory
