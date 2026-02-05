declare global {
  interface LemonSqueezyOrderData {
    type: string;
    id: string;
    attributes: Record<string, unknown>;
  }

  interface LemonSqueezyEvent {
    event: string;
    data?: LemonSqueezyOrderData;
  }

  interface LemonSqueezyUrl {
    Open: (url: string) => void;
  }

  interface LemonSqueezySetupOptions {
    eventHandler?: (event: LemonSqueezyEvent) => void;
  }

  interface LemonSqueezyAPI {
    Url: LemonSqueezyUrl;
    Setup: (options: LemonSqueezySetupOptions) => void;
  }

  const LemonSqueezy: LemonSqueezyAPI | undefined;
}

export {};
