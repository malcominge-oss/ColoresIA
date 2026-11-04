/* Conecta tu HTML existente con el servidor Python. No modifica tu diseño ni tus elementos:
   solo escucha tu botón "Detectar", lee tu canvas / input de archivo y pinta los resultados
   en un contenedor propio. AJUSTA los IDs de CONFIG a los de tu página. */
(function () {
  'use strict';

  const CONFIG = {
    ID_CANVAS: 'canvas',          // <canvas> donde se dibuja
    ID_ARCHIVO: 'archivo',        // <input type="file">
    ID_DETECTAR: 'detectar',      // botón Detectar
    ID_RESULTADOS: 'resultados',  // contenedor de resultados (si no existe, se crea uno)
    ID_METODO: 'metodo',          // <select> opcional: bfs | dfs | heuristica
    METODO_POR_DEFECTO: 'heuristica',
    COMPARAR_METODOS: true,
    API_BASE: location.protocol === 'file:' ? 'http://127.0.0.1:8000' : ''
  };

  const $ = (id, fallback) => document.getElementById(id) || (fallback ? document.querySelector(fallback) : null);
  let ultimoOrigen = null;   // 'dibujo' | 'archivo'

  function el(tag, props, ...hijos) {
    const e = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => {
      if (k === 'style') Object.assign(e.style, v);
      else if (k === 'class') e.className = v;
      else e.setAttribute(k, v);
    });
    hijos.flat().forEach(h => { if (h != null) e.append(h.nodeType ? h : document.createTextNode(String(h))); });
    return e;
  }

  function inyectarEstilos() {
    if (document.getElementById('ia-estilos')) return;
    const st = el('style', { id: 'ia-estilos' });
    st.textContent = `
      .ia-res{font:inherit;color:inherit;margin-top:1rem}
      .ia-res h3,.ia-res h4{margin:.8rem 0 .4rem}
      .ia-fila{display:grid;grid-template-columns:28px 1fr;gap:.6rem;align-items:center;padding:.45rem 0;border-bottom:1px solid rgba(128,128,128,.25)}
      .ia-mue{width:28px;height:28px;border-radius:6px;border:1px solid rgba(128,128,128,.5)}
      .ia-barra{height:8px;border-radius:4px;background:rgba(128,128,128,.2);overflow:hidden;margin:.25rem 0}
      .ia-barra>span{display:block;height:100%}
      .ia-dist{display:flex;height:22px;border-radius:6px;overflow:hidden;border:1px solid rgba(128,128,128,.4)}
      .ia-dist>span{height:100%}
      .ia-peq{font-size:.85em;opacity:.8}
      .ia-tabla{border-collapse:collapse;width:100%;font-size:.9em}
      .ia-tabla th,.ia-tabla td{border:1px solid rgba(128,128,128,.3);padding:.25rem .5rem;text-align:left}
      .ia-error{color:#c0392b;font-weight:600}`;
    document.head.appendChild(st);
  }

  function contenedor(canvas) {
    let c = $(CONFIG.ID_RESULTADOS);
    if (!c) {
      c = el('div', { id: CONFIG.ID_RESULTADOS });
      (canvas && canvas.parentElement ? canvas.parentElement : document.body).appendChild(c);
    }
    return c;
  }

  function pintarError(c, msg) {
    c.replaceChildren(el('div', { class: 'ia-res' }, el('p', { class: 'ia-error' }, msg)));
  }

  function pintar(c, r) {
    const raiz = el('div', { class: 'ia-res' });
    raiz.append(el('h3', {}, 'Colores reconocidos'));

    const filas = r.colores.map(col => el('div', { class: 'ia-fila' },
      el('div', { class: 'ia-mue', style: { background: col.hex } }),
      el('div', {},
        el('strong', {}, `${col.nombre} `),
        el('span', { class: 'ia-peq' }, `${col.hex} · RGB(${col.rgb.join(', ')}) · ${col.pixeles} px`),
        el('div', { class: 'ia-barra' }, el('span', { style: { width: col.presencia + '%', background: col.hex } })),
        el('span', {}, `Presencia ${col.presencia}% · Similitud ${col.similitud}% · ΔE ${col.delta_e} · Coste ${col.coste}`))));
    raiz.append(...filas);
    if (r.desconocido.pixeles > 0) {
      raiz.append(el('div', { class: 'ia-fila' },
        el('div', { class: 'ia-mue', style: { background: 'repeating-linear-gradient(45deg,#999,#999 4px,#ccc 4px,#ccc 8px)' } }),
        el('div', {}, el('strong', {}, 'Desconocido '),
          el('span', {}, `Presencia ${r.desconocido.presencia}% · ${r.desconocido.pixeles} px`))));
    }

    raiz.append(el('h4', {}, 'Distribución porcentual'));
    const dist = el('div', { class: 'ia-dist' });
    r.colores.forEach(col => dist.append(el('span', { title: `${col.nombre} ${col.presencia}%`, style: { width: col.presencia + '%', background: col.hex } })));
    if (r.desconocido.pixeles > 0) dist.append(el('span', { title: 'Desconocido', style: { width: r.desconocido.presencia + '%', background: '#999' } }));
    raiz.append(dist, el('p', { class: 'ia-peq' }, `Suma de presencias: ${r.suma_presencia}%`));

    const t = r.tecnico;
    raiz.append(el('h4', {}, 'Información técnica'));
    const datos = [
      ['Nodo inicial', t.nodo_inicial], ['Nodo objetivo', t.nodo_objetivo],
      ['Método de búsqueda', t.metodo_busqueda],
      ['Nodos explorados / generados', `${t.nodos_explorados} / ${t.nodos_generados}`],
      ['Rasgos extraídos', t.rasgos_extraidos.join(', ')],
      ['Coste total', t.coste_total], ['Tiempo total', `${t.tiempos_ms.total} ms`],
      ['Píxeles analizados', `${t.imagen.pixeles_validos} de ${t.imagen.pixeles_totales}`],
      ['Pesos (rgb/hsv/lab/hist)', Object.values(t.parametros.pesos).join(' / ')],
      ['Tolerancia', t.parametros.tolerancia]
    ];
    if (t.algoritmo_genetico) {
      datos.push(['Algoritmo genético', `generación ${t.algoritmo_genetico.generacion}, fitness ${t.algoritmo_genetico.fitness}`]);
    } else {
      datos.push(['Algoritmo genético', 'sin entrenar: se usan parámetros por defecto']);
    }
    raiz.append(el('table', { class: 'ia-tabla' }, el('tbody', {}, datos.map(([k, v]) => el('tr', {}, el('th', {}, k), el('td', {}, v))))));

    if (r.comparacion) {
      raiz.append(el('h4', {}, 'Comparación de algoritmos de búsqueda'));
      raiz.append(el('table', { class: 'ia-tabla' },
        el('thead', {}, el('tr', {}, ['Método', 'Nodos explorados', 'Nodos generados', 'Tiempo (ms)', 'Coste', 'Mismo resultado'].map(x => el('th', {}, x)))),
        el('tbody', {}, r.comparacion.map(f => el('tr', {},
          el('td', {}, f.metodo), el('td', {}, f.nodos_explorados), el('td', {}, f.nodos_generados),
          el('td', {}, f.tiempo_busqueda_ms), el('td', {}, f.coste_total ?? '-'),
          el('td', {}, f.mismo_resultado_que_heuristica == null ? '-' : (f.mismo_resultado_que_heuristica ? 'sí' : 'no')))))));
    }
    c.replaceChildren(raiz);
  }

  async function detectar() {
    const canvas = $(CONFIG.ID_CANVAS, 'canvas');
    const input = $(CONFIG.ID_ARCHIVO, 'input[type="file"]');
    const cont = contenedor(canvas);
    inyectarEstilos();
    cont.replaceChildren(el('p', { class: 'ia-res ia-peq' }, 'Analizando…'));

    const fd = new FormData();
    const hayArchivo = input && input.files && input.files[0];
    const origen = ultimoOrigen || (hayArchivo ? 'archivo' : 'dibujo');
    try {
      if (origen === 'archivo' && hayArchivo) {
        fd.append('file', input.files[0]);
        fd.append('origen', 'archivo');
      } else {
        if (!canvas) throw new Error('No se encontró el canvas (revisa CONFIG.ID_CANVAS).');
        const blob = await new Promise(ok => canvas.toBlob(ok, 'image/png'));
        if (!blob) throw new Error('No se pudo leer el canvas.');
        fd.append('file', blob, 'canvas.png');
        fd.append('origen', 'dibujo');
      }
      const sel = $(CONFIG.ID_METODO);
      fd.append('metodo', (sel && sel.value) || CONFIG.METODO_POR_DEFECTO);
      fd.append('comparar', CONFIG.COMPARAR_METODOS ? 'true' : 'false');

      const resp = await fetch(CONFIG.API_BASE + '/api/detectar', { method: 'POST', body: fd });
      const datos = await resp.json().catch(() => ({}));
      if (!resp.ok || !datos.ok) throw new Error(datos.error || `Error del servidor (${resp.status}).`);
      pintar(cont, datos);
    } catch (e) {
      const msg = e instanceof TypeError ? 'No se pudo conectar con el servidor Python. ¿Está iniciado?' : e.message;
      pintarError(cont, msg);
    }
  }

  function iniciar() {
    const canvas = $(CONFIG.ID_CANVAS, 'canvas');
    const input = $(CONFIG.ID_ARCHIVO, 'input[type="file"]');
    let boton = $(CONFIG.ID_DETECTAR);
    if (!boton) {
      boton = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"]'))
        .find(b => /detect/i.test(b.textContent || b.value || b.id || ''));
    }
    if (!boton) { console.warn('[IA colores] No se encontró el botón Detectar (CONFIG.ID_DETECTAR).'); return; }
    if (canvas) ['pointerdown', 'mousedown', 'touchstart'].forEach(ev =>
      canvas.addEventListener(ev, () => { ultimoOrigen = 'dibujo'; }, { passive: true }));
    if (input) input.addEventListener('change', () => { ultimoOrigen = 'archivo'; });
    boton.addEventListener('click', detectar);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', iniciar);
  else iniciar();
})();
