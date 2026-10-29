// Uso: node bench.js <url_base>   (sirve navegador/ + ort + tf + modelos con un http.server)
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ args: ['--use-gl=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage();
  await p.goto(process.argv[2] + '/bench.html');
  const r = await p.evaluate(() => window.correr());
  const med = a => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
  const out = { segundos_audio_por_ventana: r.segundos_audio_por_ventana };
  for (const [k, v] of Object.entries(r)) {
    if (typeof v === 'object' && v.tiempos) out[k] = { carga_ms: Math.round(v.carga_ms), mediana_ms_por_ventana: +med(v.tiempos).toFixed(1),
      veces_tiempo_real: +(r.segundos_audio_por_ventana * 1000 / med(v.tiempos)).toFixed(1) };
    else if (k !== 'segundos_audio_por_ventana') out[k] = v;
  }
  console.log(JSON.stringify(out, null, 1));
  await b.close();
})();
