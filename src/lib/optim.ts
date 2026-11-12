export type Surface = 'bowl' | 'ravine' | 'saddle' | 'rosenbrock'

/** Real loss surfaces, with real analytic gradients. */
export const SURFACES: Record<Surface, { f: (x: number, y: number) => number
                                         g: (x: number, y: number) => [number, number]
                                         start: [number, number]; note: string }> = {
  bowl: { f: (x, y) => x * x + y * y, g: (x, y) => [2 * x, 2 * y],
          start: [-1.6, 1.4], note: 'Round and easy. Everything works here.' },
  ravine: { f: (x, y) => 0.05 * x * x + 4 * y * y, g: (x, y) => [0.1 * x, 8 * y],
            start: [-1.8, 0.9], note: 'Steep one way, flat the other. This is what real losses look like.' },
  saddle: { f: (x, y) => x * x - y * y + 0.15 * y * y * y * y,
            g: (x, y) => [2 * x, -2 * y + 0.6 * y * y * y],
            start: [-1.5, 0.02], note: 'Flat at the centre in one direction. Plain SGD stalls here.' },
  rosenbrock: { f: (x, y) => (1 - x) ** 2 + 20 * (y - x * x) ** 2,
                g: (x, y) => [-2 * (1 - x) - 80 * x * (y - x * x), 40 * (y - x * x)],
                start: [-1.3, 1.4], note: 'A curved valley. Momentum helps, raw SGD crawls.' },
}

export type OptName = 'SGD' | 'SGD + momentum' | 'RMSProp' | 'Adam' | 'AdamW'

/** Each optimizer, written out as the update rule it actually applies. */
export function runOptimizer(name: OptName, surface: Surface, lr: number, steps: number) {
  const { g, start } = SURFACES[surface]
  let [x, y] = start
  let mx = 0, my = 0, vx = 0, vy = 0
  const path: [number, number][] = [[x, y]]
  const b1 = 0.9, b2 = 0.999, eps = 1e-8, mom = 0.9, wd = 0.01

  for (let t = 1; t <= steps; t++) {
    const [gx, gy] = g(x, y)
    if (!Number.isFinite(gx) || !Number.isFinite(gy)) break
    let ux = 0, uy = 0

    switch (name) {
      case 'SGD':
        ux = lr * gx; uy = lr * gy; break
      case 'SGD + momentum':
        mx = mom * mx + gx; my = mom * my + gy
        ux = lr * mx; uy = lr * my; break
      case 'RMSProp':
        vx = 0.9 * vx + 0.1 * gx * gx; vy = 0.9 * vy + 0.1 * gy * gy
        ux = lr * gx / (Math.sqrt(vx) + eps); uy = lr * gy / (Math.sqrt(vy) + eps); break
      case 'Adam':
      case 'AdamW': {
        mx = b1 * mx + (1 - b1) * gx; my = b1 * my + (1 - b1) * gy
        vx = b2 * vx + (1 - b2) * gx * gx; vy = b2 * vy + (1 - b2) * gy * gy
        const mhx = mx / (1 - b1 ** t), mhy = my / (1 - b1 ** t)
        const vhx = vx / (1 - b2 ** t), vhy = vy / (1 - b2 ** t)
        ux = lr * mhx / (Math.sqrt(vhx) + eps); uy = lr * mhy / (Math.sqrt(vhy) + eps)
        if (name === 'AdamW') { ux += lr * wd * x; uy += lr * wd * y }
        break
      }
    }
    x -= ux; y -= uy
    if (Math.abs(x) > 50 || Math.abs(y) > 50) { path.push([x, y]); break }
    path.push([x, y])
  }
  return path
}
