/**
 * Coordinate Transformation Utility for Camera Video Overlay
 * Accurately transforms normalized bounding boxes to screen pixels,
 * properly handling:
 * - Mobile Portrait / Landscape
 * - CSS object-fit (cover, contain, fill)
 * - Cropping and aspect ratio offsets
 * - Dynamic window / element resizing
 */
export function transformBBoxToVideoOverlay(bboxNorm, videoElement, canvasElement) {
  if (!bboxNorm || !videoElement || !canvasElement) {
    return { x: 0, y: 0, w: 0, h: 0, visible: false };
  }

  const srcW = videoElement.videoWidth;
  const srcH = videoElement.videoHeight;
  const dispW = canvasElement.width;
  const dispH = canvasElement.height;

  if (!srcW || !srcH || !dispW || !dispH) {
    return { x: 0, y: 0, w: 0, h: 0, visible: false };
  }

  // Get computed object-fit style
  const videoStyle = window.getComputedStyle(videoElement);
  const objectFit = videoStyle.objectFit || 'cover';

  const srcRatio = srcW / srcH;
  const dispRatio = dispW / dispH;

  let renderW, renderH, offsetX, offsetY;

  if (objectFit === 'cover') {
    if (dispRatio > srcRatio) {
      // Display is wider than video -> Video scaled to fit width, cropped vertically
      renderW = dispW;
      renderH = dispW / srcRatio;
      offsetX = 0;
      offsetY = (dispH - renderH) / 2;
    } else {
      // Display is taller than video -> Video scaled to fit height, cropped horizontally
      renderH = dispH;
      renderW = dispH * srcRatio;
      offsetX = (dispW - renderW) / 2;
      offsetY = 0;
    }
  } else if (objectFit === 'contain') {
    if (dispRatio > srcRatio) {
      // Video letterboxed horizontally (bars on left & right)
      renderH = dispH;
      renderW = dispH * srcRatio;
      offsetX = (dispW - renderW) / 2;
      offsetY = 0;
    } else {
      // Video letterboxed vertically (bars on top & bottom)
      renderW = dispW;
      renderH = dispW / srcRatio;
      offsetX = 0;
      offsetY = (dispH - renderH) / 2;
    }
  } else {
    // Fill / stretch
    renderW = dispW;
    renderH = dispH;
    offsetX = 0;
    offsetY = 0;
  }

  // Transform normalized coords (0..1) to rendered video space
  const x1 = offsetX + bboxNorm.x1 * renderW;
  const y1 = offsetY + bboxNorm.y1 * renderH;
  const x2 = offsetX + bboxNorm.x2 * renderW;
  const y2 = offsetY + bboxNorm.y2 * renderH;

  return {
    x: x1,
    y: y1,
    w: Math.max(1, x2 - x1),
    h: Math.max(1, y2 - y1),
    visible: true
  };
}

export default transformBBoxToVideoOverlay;
