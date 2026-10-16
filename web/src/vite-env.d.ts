/// <reference types="vite/client" />

declare namespace JSX {
  interface IntrinsicElements {
    'model-viewer': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
      src?: string;
      alt?: string;
      'camera-controls'?: boolean | string;
      'auto-rotate'?: boolean | string;
      poster?: string;
      loading?: string;
      exposure?: string;
      'shadow-intensity'?: string;
    };
  }
}
