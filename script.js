
const lienzo = document.getElementById("lienzo");
const ctx = lienzo.getContext("2d");

const limpiar = document.getElementById("limpiar");
const imagen = document.getElementById("imagen");
const detectar = document.getElementById("detectar");

const nombreColor = document.getElementById("nombreColor");
const muestraColor = document.getElementById("muestraColor");
const resultado = document.getElementById("resultado");

let colorActual = "#ff0000";
let dibujando = false;

ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, lienzo.width, lienzo.height);

document.querySelectorAll(".color").forEach(boton => {
    boton.addEventListener("click", () => {
        document.querySelectorAll(".color").forEach(b => {
            b.classList.remove("seleccionado");
        });

        boton.classList.add("seleccionado");
        colorActual = boton.dataset.color;
    });
});

const obtenerPosicion = (evento) => {
    const rect = lienzo.getBoundingClientRect();

    return {
        x: (evento.clientX - rect.left) * (lienzo.width / rect.width),
        y: (evento.clientY - rect.top) * (lienzo.height / rect.height)
    };
};

lienzo.addEventListener("pointerdown", evento => {
    dibujando = true;
    lienzo.setPointerCapture(evento.pointerId);

    const posicion = obtenerPosicion(evento);

    ctx.beginPath();
    ctx.moveTo(posicion.x, posicion.y);
    ctx.lineTo(posicion.x + 0.1, posicion.y + 0.1);
    ctx.strokeStyle = colorActual;
    ctx.lineWidth = 35;
    ctx.lineCap = "round";
    ctx.stroke();
});

lienzo.addEventListener("pointermove", evento => {
    if (!dibujando) return;

    const posicion = obtenerPosicion(evento);

    ctx.lineTo(posicion.x, posicion.y);
    ctx.strokeStyle = colorActual;
    ctx.lineWidth = 35;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
});

lienzo.addEventListener("pointerup", () => {
    dibujando = false;
    ctx.closePath();
});

lienzo.addEventListener("pointercancel", () => {
    dibujando = false;
    ctx.closePath();
});

limpiar.addEventListener("click", () => {
    ctx.clearRect(0, 0, lienzo.width, lienzo.height);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, lienzo.width, lienzo.height);

    resultado.style.display = "none";
});

imagen.addEventListener("change", evento => {
    const archivo = evento.target.files[0];

    if (!archivo) return;

    const lector = new FileReader();

    lector.onload = e => {
        const img = new Image();

        img.onload = () => {
            ctx.clearRect(0, 0, lienzo.width, lienzo.height);

            ctx.fillStyle = "#ffffff";
            ctx.fillRect(0, 0, lienzo.width, lienzo.height);

            const escala = Math.min(
                lienzo.width / img.width,
                lienzo.height / img.height
            );

            const ancho = img.width * escala;
            const alto = img.height * escala;

            const x = (lienzo.width - ancho) / 2;
            const y = (lienzo.height - alto) / 2;

            ctx.drawImage(img, x, y, ancho, alto);

            resultado.style.display = "none";
        };

        img.src = e.target.result;
    };

    lector.readAsDataURL(archivo);
});

detectar.addEventListener("click", () => {
    const centroX = Math.floor(lienzo.width / 2);
    const centroY = Math.floor(lienzo.height / 2);

    const pixel = ctx.getImageData(centroX, centroY, 1, 1).data;

    const r = pixel[0];
    const g = pixel[1];
    const b = pixel[2];

    const hex = "#" + [r, g, b]
        .map(valor => valor.toString(16).padStart(2, "0"))
        .join("");

    const colores = {
        rojo: [255, 0, 0],
        naranja: [255, 127, 0],
        amarillo: [255, 255, 0],
        verde: [0, 255, 0],
        azul: [0, 0, 255],
        indigo: [75, 0, 130],
        violeta: [148, 0, 211],
        blanco: [255, 255, 255],
        negro: [0, 0, 0],
        rosa: [255, 105, 180]
    };

    let colorMasCercano = "";
    let distanciaMinima = Infinity;

    for (const [nombre, rgb] of Object.entries(colores)) {
        const distancia = Math.sqrt(
            Math.pow(r - rgb[0], 2) +
            Math.pow(g - rgb[1], 2) +
            Math.pow(b - rgb[2], 2)
        );

        if (distancia < distanciaMinima) {
            distanciaMinima = distancia;
            colorMasCercano = nombre;
        }
    }

    nombreColor.textContent =
        colorMasCercano.charAt(0).toUpperCase() +
        colorMasCercano.slice(1) + " (" + hex + ")";

    muestraColor.style.backgroundColor = hex;

    resultado.style.display = "block";
});
